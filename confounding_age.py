#!/usr/bin/env python3
"""
Confounding-aware regression with Transformer models (QLoRA).
Models: sdadas/stella-pl, allegro/herbert-large-cased
Approaches: Feature Concatenation, Multi-Task, Adversarial (GRL), Residual Learning, Special Tokens
"""

import argparse
import os
import warnings
import json
import csv
import pickle
import shutil
import itertools
import math
import re
from datetime import timedelta
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from huggingface_hub.errors import RepositoryNotFoundError
try:
    from huggingface_hub.errors import GatedRepoError
except ImportError:
    GatedRepoError = RepositoryNotFoundError

if not hasattr(torch, "float8_e8m0fnu") and hasattr(torch, "float8_e4m3fn"):
    torch.float8_e8m0fnu = torch.float8_e4m3fn

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
import gc

warnings.filterwarnings("ignore")

# ============================================================
# Metrics
# ============================================================

def nonparametric_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.median(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


def spearman_rho_squared(y_true, y_pred):
    rho, _ = spearmanr(y_true, y_pred)
    return rho ** 2


def mape(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def compute_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "NP_R2": nonparametric_r2(y_true, y_pred),
        "Spearman_rho2": spearman_rho_squared(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }


def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    vals = np.array(vals)
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(vals, [alpha * 100, (1 - alpha) * 100])
    return np.mean(vals), lo, hi


TOKEN_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def get_unique_token_count(text):
    if not isinstance(text, str):
        return 0
    tokens = TOKEN_PATTERN.findall(text.lower())
    return len(set(tokens))


def build_ridge_baseline_features(texts, confounders_encoded):
    unique_token_counts = np.asarray(
        [get_unique_token_count(text) for text in texts], dtype=np.float32
    )
    group_feature = np.asarray(confounders_encoded, dtype=np.float32)
    return np.column_stack([group_feature, unique_token_counts]).astype(np.float32)


def run_ridge_baseline_cv(texts, targets, confounders_encoded, skf, output_dir,
                          target_cols, n_bootstrap=1000, rank=0):
    """Run a simple Ridge baseline with GRUPA + unique token count features."""
    features = build_ridge_baseline_features(texts, confounders_encoded)
    fold_indices = list(skf.split(np.zeros(len(targets)), confounders_encoded))
    metric_fns = {
        "MAE": lambda yt, yp: mean_absolute_error(yt, yp),
        "R2": lambda yt, yp: r2_score(yt, yp),
        "NP_R2": nonparametric_r2,
        "Spearman_rho2": spearman_rho_squared,
        "MedAE": lambda yt, yp: median_absolute_error(yt, yp),
        "MAPE": mape,
    }

    baseline_rows = []
    for target_idx, target_col in enumerate(target_cols):
        y_true = targets[:, target_idx]
        oof_preds = np.zeros(len(y_true), dtype=np.float32)

        for train_idx, val_idx in fold_indices:
            baseline_model = Pipeline([
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0)),
            ])
            baseline_model.fit(features[train_idx], y_true[train_idx])
            oof_preds[val_idx] = baseline_model.predict(features[val_idx]).astype(np.float32)

        result_row = {
            "model": "ridge_baseline",
            "approach": "baseline",
            "loss": "ridge",
            "target": target_col,
        }
        for metric_name, metric_fn in metric_fns.items():
            mean_val, lo, hi = bootstrap_ci(
                y_true, oof_preds, metric_fn, n_boot=n_bootstrap
            )
            result_row[f"{metric_name}_mean"] = mean_val
            result_row[f"{metric_name}_ci"] = (hi - lo) / 2

        baseline_rows.append(result_row)

        if rank == 0:
            print(
                f"    [ridge baseline] {target_col}: "
                f"MAE={result_row['MAE_mean']:.4f}, "
                f"R2={result_row['R2_mean']:.4f}, "
                f"Spearman_rho2={result_row['Spearman_rho2_mean']:.4f}"
            )

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_path = os.path.join(output_dir, "ridge_baseline_results.csv")
    if rank == 0:
        baseline_df.to_csv(baseline_path, index=False)
        print(f"    Saved Ridge baseline metrics to {baseline_path}")
    return baseline_df


# ============================================================
# Dataset
# ============================================================

EMPTY_TEXT_PLACEHOLDER = "[BRAK TEKSTU]"


def resolve_model_name(model_name):
    """Resolve short aliases to full HF model IDs."""
    aliases = {
        "stella-pl-retrieval-8k": "sdadas/stella-pl-retrieval-8k",
        "polish-roberta-8k": "sdadas/polish-roberta-8k",
    }
    return aliases.get(model_name.strip(), model_name.strip())


def is_model_resolvable(model_name, token):
    """Return True when the model config can be fetched from Hugging Face."""
    try:
        AutoConfig.from_pretrained(model_name, token=token)
        return True
    except (RepositoryNotFoundError, GatedRepoError, OSError):
        return False


def parse_model_int_map(spec):
    """Parse model integer mapping from 'model=val,model=val' string."""
    mapping = {}
    if not spec:
        return mapping
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            continue
        name, val = item.split("=", 1)
        name = name.strip()
        try:
            mapping[name] = int(val.strip())
        except ValueError:
            continue
    return mapping


def get_effective_max_len(model_name, tokenizer, model_len_map, default_len):
    """Return model-specific max length capped by tokenizer limits when available."""
    desired = model_len_map.get(model_name, default_len)
    tok_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_limit, int) and tok_limit > 0 and tok_limit < 10**6:
        return min(desired, tok_limit)
    return desired


def infer_lora_target_modules(model):
    """Infer LoRA target module leaf names from available Linear layers."""
    linear_leaf_names = sorted({
        name.split(".")[-1]
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    })

    # Most common attention projection names across BERT/RoBERTa/LLaMA-like variants.
    preferred = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "query", "key", "value", "dense",
        "query_key_value", "c_attn", "qkv", "Wqkv", "out_proj",
    ]
    selected = [name for name in preferred if name in linear_leaf_names]
    if selected:
        return selected

    # Fallback: use attention-related linear leaf names only.
    attn_leaf_names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        lname = name.lower()
        if "attn" in lname or "attention" in lname:
            leaf = name.split(".")[-1]
            if leaf not in attn_leaf_names:
                attn_leaf_names.append(leaf)
    if attn_leaf_names:
        return attn_leaf_names

    # Last resort: at least one valid linear module to avoid hard crash.
    if linear_leaf_names:
        return [linear_leaf_names[0]]
    raise ValueError("No nn.Linear modules found to target for LoRA.")


def tokenize_with_strategy(tokenizer, text, max_len, strategy="head"):
    """Tokenize with head / tail / head_tail truncation."""
    def _resolve_special_id(primary_name, fallback_names=()):
        token_id = getattr(tokenizer, primary_name, None)
        if token_id is not None:
            return token_id
        for fallback_name in fallback_names:
            token_id = getattr(tokenizer, fallback_name, None)
            if token_id is not None:
                return token_id
        return None

    if strategy == "tail":
        enc = tokenizer(
            text, max_length=max_len, padding="max_length",
            truncation=True, truncation_side="left", return_tensors="pt"
        )
    elif strategy == "head_tail":
        # Keep first head_n tokens + last tail_n tokens
        head_n = max_len // 4          # 128 out of 512
        tail_n = max_len - head_n - 2  # 382; 2 for [CLS]/[SEP]
        tokens = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )["input_ids"]
        if len(tokens) <= max_len - 2:
            enc = tokenizer(
                text, max_length=max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )
        else:
            merged = tokens[:head_n] + tokens[-tail_n:]
            cls_id = _resolve_special_id("cls_token_id", ("bos_token_id",))
            sep_id = _resolve_special_id("sep_token_id", ("eos_token_id",))
            pad_id = _resolve_special_id("pad_token_id", ("eos_token_id", "unk_token_id"))
            if pad_id is None:
                pad_id = 0

            ids = list(merged)
            if cls_id is not None:
                ids = [cls_id] + ids
            if sep_id is not None:
                ids = ids + [sep_id]

            if len(ids) > max_len:
                ids = ids[:max_len]

            pad_len = max_len - len(ids)
            attention_mask = [1] * len(ids) + [0] * pad_len
            ids = ids + [pad_id] * pad_len
            enc = {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }
    else:  # head (default)
        enc = tokenizer(
            text, max_length=max_len, padding="max_length",
            truncation=True, return_tensors="pt"
        )
    return enc


class TextRegressionDataset(Dataset):
    def __init__(self, texts, targets, confounders, tokenizer, max_len=512,
                 confounder_ohe=None, approach="baseline", special_token_map=None,
                 truncation_strategy="head", text_indices=None, confounder_reg=None):
        self.texts = texts
        self.targets = targets
        self.confounders = confounders
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.confounder_ohe = confounder_ohe
        self.approach = approach
        self.special_token_map = special_token_map
        self.truncation_strategy = truncation_strategy
        self.text_indices = text_indices
        self.confounder_reg = confounder_reg

    def __len__(self):
        return len(self.text_indices) if self.text_indices is not None else len(self.texts)

    def _resolve_text_index(self, idx):
        if self.text_indices is None:
            return idx
        return int(self.text_indices[idx])

    def __getitem__(self, idx):
        source_idx = self._resolve_text_index(idx)
        text = str(self.texts[source_idx]).strip() or EMPTY_TEXT_PLACEHOLDER
        if self.approach == "special_tokens" and self.special_token_map is not None:
            conf_val = self.confounders[source_idx]
            token = self.special_token_map.get(conf_val, "")
            text = f"{token} {text}"

        enc = tokenize_with_strategy(self.tokenizer, text, self.max_len,
                                     self.truncation_strategy)
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["targets"] = torch.tensor(self.targets[source_idx], dtype=torch.float32)
        item["confounder_idx"] = torch.tensor(self.confounders[source_idx], dtype=torch.long) \
            if isinstance(self.confounders[source_idx], (int, np.integer)) else torch.tensor(0)

        if self.confounder_ohe is not None:
            ohe_vec = self.confounder_ohe[source_idx]
            item["confounder_ohe"] = torch.tensor(ohe_vec, dtype=torch.float32)

        if self.confounder_reg is not None:
            item["confounder_reg"] = torch.tensor(
                self.confounder_reg[source_idx], dtype=torch.float32
            )

        return item


# ============================================================
# Gradient Reversal Layer
# ============================================================

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        # Stored as a buffer (not a parameter) so it can be mutated during
        # training (Ganin-style warmup) and stays on the right device.
        self.register_buffer("alpha", torch.tensor(float(alpha)), persistent=False)

    def set_alpha(self, value):
        with torch.no_grad():
            self.alpha.fill_(float(value))

    def forward(self, x):
        return GradientReversalFunction.apply(x, float(self.alpha.item()))


# ============================================================
# Model Heads
# ============================================================

class FeatureConcatHead(nn.Module):
    """Approach 1 (legacy concat): Linear over [cls || ohe]."""
    def __init__(self, hidden_size, n_confounder_dims, n_targets):
        super().__init__()
        self.regressor = nn.Linear(hidden_size + n_confounder_dims, n_targets)

    def forward(self, cls_output, confounder_ohe):
        combined = torch.cat([cls_output, confounder_ohe], dim=-1)
        return self.regressor(combined)


class AdditiveFeatureConcatHead(nn.Module):
    """Approach 1 (additive two-tower): y = text_branch(cls) + conf_branch(ohe).

    Decouples the text and confounder contributions so that (a) the conf branch
    is not crushed by weight decay on the large [CLS] tower (HerBERT issue) and
    (b) per-sample conf-dropout during training prevents the model from
    over-relying on the tag when the encoder is weak (Stella issue).

    The conf_branch is a small MLP so it can express non-linear group effects
    (useful in 'ordinal' confounder mode where the OHE is a scalar).
    """
    def __init__(self, hidden_size, n_confounder_dims, n_targets,
                 conf_hidden=64, conf_dropout=0.2):
        super().__init__()
        self.text_branch = nn.Linear(hidden_size, n_targets)
        self.conf_dropout = nn.Dropout(p=conf_dropout)
        self.conf_branch = nn.Sequential(
            nn.Linear(n_confounder_dims, conf_hidden),
            nn.GELU(),
            nn.Linear(conf_hidden, n_targets),
        )

    def forward(self, cls_output, confounder_ohe):
        text_pred = self.text_branch(cls_output)
        # Per-sample dropout on the OHE forces the text branch to carry signal
        # even when the tag is masked.
        conf_in = self.conf_dropout(confounder_ohe)
        conf_pred = self.conf_branch(conf_in)
        return text_pred + conf_pred


class MultiTaskHead(nn.Module):
    """Approach 2: Regression + confounder prediction (class or ordinal value)."""
    def __init__(self, hidden_size, n_targets, confounder_out_dim):
        super().__init__()
        self.regressor = nn.Linear(hidden_size, n_targets)
        self.classifier = nn.Linear(hidden_size, confounder_out_dim)

    def forward(self, cls_output):
        reg_out = self.regressor(cls_output)
        cls_out = self.classifier(cls_output)
        return reg_out, cls_out


class AdversarialHead(nn.Module):
    """Approach 3: Regression + GRL + confounder predictor (class or ordinal value).

    The discriminator is a wider MLP than the toy 128-d version so it can
    actually saturate the encoder representation (otherwise the GRL pressure on
    the encoder is negligible and there is no confounder removal).
    """
    def __init__(self, hidden_size, n_targets, confounder_out_dim, alpha=1.0,
                 disc_hidden=512, disc_dropout=0.1):
        super().__init__()
        self.regressor = nn.Linear(hidden_size, n_targets)
        self.grl = GradientReversalLayer(alpha=alpha)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, disc_hidden),
            nn.GELU(),
            nn.Dropout(p=disc_dropout),
            nn.Linear(disc_hidden, disc_hidden // 2),
            nn.GELU(),
            nn.Dropout(p=disc_dropout),
            nn.Linear(disc_hidden // 2, confounder_out_dim),
        )

    def set_alpha(self, value):
        self.grl.set_alpha(value)

    def forward(self, cls_output):
        reg_out = self.regressor(cls_output)
        reversed_features = self.grl(cls_output)
        cls_out = self.classifier(reversed_features)
        return reg_out, cls_out


class ResidualHead(nn.Module):
    """Approach 4: Predict residuals only."""
    def __init__(self, hidden_size, n_targets):
        super().__init__()
        self.regressor = nn.Linear(hidden_size, n_targets)

    def forward(self, cls_output):
        return self.regressor(cls_output)


class SpecialTokenHead(nn.Module):
    """Approach 5: Standard regression head (token already in input)."""
    def __init__(self, hidden_size, n_targets):
        super().__init__()
        self.regressor = nn.Linear(hidden_size, n_targets)

    def forward(self, cls_output):
        return self.regressor(cls_output)


def is_stella_model_name(model_name):
    return model_name == "sdadas/stella-pl" or model_name.endswith("/stella-pl")


def unfreeze_top_encoder_layers(model, num_layers):
    """Unfreeze the last few encoder blocks plus normalization layers when present."""
    if num_layers is None or num_layers <= 0:
        return

    candidate_paths = [
        "base_model.encoder.layer",
        "encoder.layer",
        "bert.encoder.layer",
        "roberta.encoder.layer",
        "model.encoder.layer",
        "transformer.layer",
        "encoder.block",
    ]

    layer_list = None
    for path in candidate_paths:
        try:
            candidate = model.get_submodule(path)
        except (AttributeError, ValueError):
            continue
        if hasattr(candidate, "__len__") and len(candidate) > 0:
            layer_list = candidate
            break

    if layer_list is not None:
        for block in list(layer_list)[-int(num_layers):]:
            for param in block.parameters():
                param.requires_grad = True

    for name, param in model.named_parameters():
        lname = name.lower()
        if any(token in lname for token in ("layernorm", "layer_norm", "final_layer_norm", "pooler")):
            if "encoder" in lname or "base_model" in lname or "transformer" in lname:
                param.requires_grad = True


def build_confounder_scalar(confounder_levels, confounders_encoded):
    """Map an (ordered) confounder to a standardized continuous scalar.

    The confounder reflects a quantifiable, monotonic property rather than an
    unordered category. We interpret each level as a numeric value when possible
    (e.g. "40.0", "85.0") and otherwise fall back to its ordinal rank, then
    z-score normalize. Returns per-sample scalars and per-level scalars (the
    latter is used to build counterfactuals in the metrics stage).
    """
    try:
        level_values = np.array([float(lvl) for lvl in confounder_levels], dtype=np.float64)
    except (TypeError, ValueError):
        level_values = np.arange(len(confounder_levels), dtype=np.float64)

    sample_values = level_values[confounders_encoded]
    mean = float(sample_values.mean())
    std = float(sample_values.std())
    if std < 1e-8:
        std = 1.0
    sample_scalar = ((sample_values - mean) / std).astype(np.float32)
    level_scalar = ((level_values - mean) / std).astype(np.float32)
    return sample_scalar, level_scalar


def build_residual_features(confounder_feat, confounder_mode, degree=3):
    """Features for the residual Ridge model.

    In ordinal mode the confounder is a single scalar; we expand it with
    polynomial terms so the (monotonic but non-linear) relationship between the
    confounder and the target can be captured. In categorical mode we keep the
    one-hot matrix unchanged.
    """
    if confounder_mode == "ordinal":
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        return poly.fit_transform(confounder_feat).astype(np.float32)
    return confounder_feat


class ProjectionNeck(nn.Module):
    """Zero-initialized residual MLP applied to the pooled embedding.

    LayerNorm -> Linear(hidden -> expansion) -> GELU -> Dropout ->
    Linear(expansion -> hidden), added back to the (normalized) input.
    The second Linear is zero-initialized so the block starts as an identity
    transform (after the LayerNorm) and cannot erase the pooled signal at a low
    learning rate; it gradually learns extra capacity during training. Output
    dimensionality equals hidden_size, so the downstream task head is unchanged.
    """
    def __init__(self, hidden_size, expansion_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, expansion_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(expansion_dim, hidden_size)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        normed = self.norm(x)
        residual = self.fc2(self.dropout(self.act(self.fc1(normed))))
        return normed + residual


# ============================================================
# Combined Model
# ============================================================

class ConfoundingModel(nn.Module):
    def __init__(self, model_name, approach, n_targets, n_confounder_dims,
                 n_confounder_classes, grl_alpha=1.0, lora_target_modules=None,
                 stella_lora_r=16, stella_lora_alpha=32, stella_lora_dropout=0.1,
                 stella_unfreeze_last_n_layers=0,
                 herbert_lora_r=16, herbert_lora_alpha=32, herbert_lora_dropout=0.1,
                 herbert_unfreeze_last_n_layers=0, pooling_mode=None,
                 confounder_mode="categorical", head_hidden_dim=0,
                 head_dropout=0.0, head_l2_normalize=False,
                 feature_concat_mode="additive",
                 feature_concat_conf_hidden=64,
                 feature_concat_conf_dropout=0.2,
                 use_group_bias=True,
                 special_tokens_pool_extra=True,
                 special_tokens_pool_k=4):
        super().__init__()
        self.approach = approach
        self.model_name = model_name
        self.confounder_mode = confounder_mode
        self.feature_concat_mode = feature_concat_mode
        self.special_tokens_pool_extra = bool(special_tokens_pool_extra)
        self.special_tokens_pool_k = int(special_tokens_pool_k)
        config = AutoConfig.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))
        model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                config=config,
                token=os.environ.get("HF_TOKEN"),
                low_cpu_mem_usage=True,
                offload_state_dict=True,
                torch_dtype=model_dtype,
            )
        except TypeError:
            self.encoder = AutoModel.from_pretrained(
                model_name,
                config=config,
                token=os.environ.get("HF_TOKEN"),
                low_cpu_mem_usage=True,
                offload_state_dict=True,
                torch_dtype=model_dtype,
            )
        hidden_size = config.hidden_size

        if lora_target_modules is None or len(lora_target_modules) == 0:
            lora_target_modules = infer_lora_target_modules(self.encoder)
        print(f"  LoRA target modules: {lora_target_modules}")

        is_stella = is_stella_model_name(model_name)
        is_herbert = "herbert-large-cased" in model_name
        lora_rank = stella_lora_r if is_stella else (herbert_lora_r if is_herbert else 16)
        lora_alpha = stella_lora_alpha if is_stella else (herbert_lora_alpha if is_herbert else 32)
        lora_dropout = stella_lora_dropout if is_stella else (herbert_lora_dropout if is_herbert else 0.1)

        # Pooling strategy: Stella is a sentence-embedding model trained with mean
        # pooling, so CLS-token pooling yields a weak representation. Default Stella
        # to mean pooling; keep CLS for encoder models like HerBERT unless overridden.
        if pooling_mode in ("mean", "cls"):
            self.pooling = pooling_mode
        else:
            self.pooling = "mean" if is_stella else "cls"

        # Apply QLoRA
        lora_config = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.encoder = get_peft_model(self.encoder, lora_config)

        if is_stella and stella_unfreeze_last_n_layers > 0:
            unfreeze_top_encoder_layers(self.encoder, stella_unfreeze_last_n_layers)
        if is_herbert and herbert_unfreeze_last_n_layers > 0:
            unfreeze_top_encoder_layers(self.encoder, herbert_unfreeze_last_n_layers)

        # Memory optimization for long contexts (important on T4).
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        if hasattr(self.encoder, "config") and hasattr(self.encoder.config, "use_cache"):
            self.encoder.config.use_cache = False

        # Optional projection "neck" applied to the pooled embedding before the
        # task head. Embedding models such as Stella tend to underfit when the
        # head is a single linear layer over a mostly-frozen encoder. The neck is
        # a zero-initialized residual MLP (see ProjectionNeck): it starts as a
        # near-identity transform so it cannot erase the pooled signal at a low
        # learning rate, then gradually learns extra capacity. Its output dim
        # equals hidden_size, so the task head is unchanged. Disabled (Identity)
        # when head_hidden_dim <= 0 so encoder models like HerBERT keep their
        # original linear head.
        self.l2_normalize = bool(head_l2_normalize)
        if head_hidden_dim and head_hidden_dim > 0:
            self.neck = ProjectionNeck(hidden_size, head_hidden_dim, dropout=head_dropout)
        else:
            self.neck = nn.Identity()

        # For 'special_tokens' approach we (optionally) augment the pooled vector
        # with the mean of the first K real-token hidden states (immediately
        # after [CLS]) where the prepended per-group marker lands. This gives the
        # head a dedicated per-group channel and unblocks nonzero CATE.
        self.head_extra_dim = 0
        if approach == "special_tokens" and self.special_tokens_pool_extra:
            self.head_extra_dim = hidden_size
        head_input_dim = hidden_size + self.head_extra_dim

        # In ordinal mode the confounder is predicted as a single continuous
        # value; in categorical mode it is predicted as one of n classes.
        aux_out_dim = 1 if confounder_mode == "ordinal" else n_confounder_classes
        if approach == "feature_concat":
            if feature_concat_mode == "additive":
                self.head = AdditiveFeatureConcatHead(
                    head_input_dim, n_confounder_dims, n_targets,
                    conf_hidden=feature_concat_conf_hidden,
                    conf_dropout=feature_concat_conf_dropout,
                )
            else:
                self.head = FeatureConcatHead(head_input_dim, n_confounder_dims, n_targets)
        elif approach == "multitask":
            self.head = MultiTaskHead(head_input_dim, n_targets, aux_out_dim)
        elif approach == "adversarial":
            self.head = AdversarialHead(head_input_dim, n_targets, aux_out_dim, grl_alpha)
        elif approach == "residual":
            self.head = ResidualHead(head_input_dim, n_targets)
        elif approach == "special_tokens":
            self.head = SpecialTokenHead(head_input_dim, n_targets)
        else:
            raise ValueError(f"Unknown approach: {approach}")

        # Per-group additive bias. Acts as a learnable group offset and is the
        # cheapest correction to the monotonic shrinkage observed when the head
        # has no other way to model per-group means. Disabled for 'residual'
        # (the Ridge baseline already removes the per-group mean) and for
        # 'feature_concat' (the conf branch / OHE already carries that signal).
        if use_group_bias and approach not in ("residual", "feature_concat"):
            self.group_bias = nn.Embedding(n_confounder_classes, n_targets)
            nn.init.zeros_(self.group_bias.weight)
            self._group_bias_enabled = True
        else:
            self.group_bias = None
            self._group_bias_enabled = False

        if os.environ.get("RANK", "0") == "0":
            if isinstance(self.neck, nn.Identity):
                neck_state = "off"
            else:
                neck_state = f"on(hidden={head_hidden_dim}, dropout={head_dropout})"
            n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.parameters())
            extras = []
            if approach == "feature_concat":
                extras.append(f"fc_mode={feature_concat_mode}")
            if approach == "special_tokens" and self.special_tokens_pool_extra:
                extras.append(f"st_pool=cls+first{self.special_tokens_pool_k}")
            if self._group_bias_enabled:
                extras.append("group_bias=on")
            extras_str = (", " + ", ".join(extras)) if extras else ""
            print(f"  [model] pooling={self.pooling}, neck={neck_state}, "
                  f"l2_normalize={self.l2_normalize}{extras_str}, "
                  f"trainable_params={n_trainable:,}/{n_total:,}")

    def init_group_bias_from_means(self, per_group_means):
        """Initialize the per-group bias from a (n_classes, n_targets) array.

        Call once after each fold split with the per-group mean of the train
        targets (in whatever scale the model is trained on, i.e. standardized
        if target standardization is active, or residuals if residual approach
        is active). The head still starts at the global mean / zero, so adding
        these biases anchors predictions on the correct per-group level from
        the very first epoch and removes monotonic shrinkage.
        """
        if self.group_bias is None:
            return
        weight = self.group_bias.weight
        arr = torch.as_tensor(per_group_means, dtype=weight.dtype, device=weight.device)
        if arr.shape != weight.shape:
            return
        with torch.no_grad():
            weight.copy_(arr)

    def set_grl_alpha(self, value):
        """Update the GRL alpha buffer (no-op for non-adversarial heads)."""
        head = self.head
        if hasattr(head, "set_alpha"):
            head.set_alpha(value)

    def _pool(self, last_hidden_state, attention_mask):
        """Pool token representations into a single vector per sequence."""
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
            summed = (last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            return summed / counts
        return last_hidden_state[:, 0, :]  # [CLS]

    def _pool_first_tokens(self, last_hidden_state, attention_mask, k):
        """Mean over positions [1..k] (right after [CLS]) where the special
        per-group marker prepended in the dataset lands. Uses the attention
        mask so short sequences without k real tokens are not biased by pads.
        """
        if k <= 0:
            return None
        # Positions 1..k (0 is [CLS]); cap to actual seq length.
        seq_len = last_hidden_state.size(1)
        end = min(1 + k, seq_len)
        if end <= 1:
            return last_hidden_state[:, 0, :]
        slice_hidden = last_hidden_state[:, 1:end, :]
        slice_mask = attention_mask[:, 1:end].unsqueeze(-1).to(slice_hidden.dtype)
        summed = (slice_hidden * slice_mask).sum(dim=1)
        counts = slice_mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(self, input_ids, attention_mask, confounder_ohe=None,
                confounder_idx=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        head_dtype = next(self.head.parameters()).dtype
        if pooled.dtype != head_dtype:
            pooled = pooled.to(dtype=head_dtype)
        pooled = self.neck(pooled)
        if self.l2_normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)

        # Optional special-token pooling: concatenate mean of first K post-CLS
        # token states so the head has a dedicated per-group channel.
        if self.approach == "special_tokens" and self.special_tokens_pool_extra:
            st_pool = self._pool_first_tokens(
                outputs.last_hidden_state, attention_mask, self.special_tokens_pool_k
            )
            if st_pool.dtype != head_dtype:
                st_pool = st_pool.to(dtype=head_dtype)
            pooled = torch.cat([pooled, st_pool], dim=-1)

        if self.approach == "feature_concat":
            if confounder_ohe is not None and confounder_ohe.dtype != head_dtype:
                confounder_ohe = confounder_ohe.to(dtype=head_dtype)
            reg_out = self.head(pooled, confounder_ohe)
            cls_out = None
        elif self.approach in ("multitask", "adversarial"):
            reg_out, cls_out = self.head(pooled)
        else:
            reg_out = self.head(pooled)
            cls_out = None

        if self._group_bias_enabled and confounder_idx is not None:
            bias = self.group_bias(confounder_idx)
            if bias.dtype != reg_out.dtype:
                bias = bias.to(dtype=reg_out.dtype)
            reg_out = reg_out + bias

        return reg_out, cls_out

    def get_cls_embeddings(self, input_ids, attention_mask):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self._pool(outputs.last_hidden_state, attention_mask).float().cpu().numpy()


# ============================================================
# Training
# ============================================================

def train_epoch(model, dataloader, optimizer, loss_fn, approach, device,
                n_confounder_classes=None, cls_weight=0.5,
                accum_steps=1, use_amp=False, scaler=None, scheduler=None,
                max_steps=None, confounder_mode="categorical",
                grl_schedule="gan", grl_alpha_max=1.0,
                global_step_state=None, total_train_steps=None):
    """Train one epoch.

    For 'adversarial' with grl_schedule='gan' the GRL alpha follows the
    Ganin-Lempitsky warmup alpha(p) = grl_alpha_max * (2/(1+exp(-10*p)) - 1)
    where p = current_step / total_train_steps. With grl_schedule='fixed' the
    alpha set at model-construction time is kept.

    global_step_state is a single-key mutable dict {'step': int} shared across
    epochs so progress p monotonically increases over the whole run.
    """
    model.train()
    total_loss = 0
    optimizer.zero_grad(set_to_none=True)
    use_grl_schedule = (
        approach == "adversarial"
        and grl_schedule == "gan"
        and total_train_steps is not None
        and total_train_steps > 0
    )
    for step_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)
        confounder_ohe = batch.get("confounder_ohe")
        if confounder_ohe is not None:
            confounder_ohe = confounder_ohe.to(device)
        confounder_idx = batch["confounder_idx"].to(device)
        confounder_reg = batch.get("confounder_reg")
        if confounder_reg is not None:
            confounder_reg = confounder_reg.to(device)

        if use_grl_schedule:
            cur_step = int(global_step_state.get("step", 0)) if global_step_state else step_idx
            progress = min(1.0, max(0.0, cur_step / float(total_train_steps)))
            alpha_val = grl_alpha_max * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
            core = model.module if isinstance(model, DDP) else model
            core.set_grl_alpha(alpha_val)

        with torch.cuda.amp.autocast(enabled=use_amp):
            reg_out, cls_out = model(input_ids, attention_mask,
                                     confounder_ohe=confounder_ohe,
                                     confounder_idx=confounder_idx)

            reg_loss = loss_fn(reg_out, targets)

            if approach in ("multitask", "adversarial") and cls_out is not None:
                if confounder_mode == "ordinal" and confounder_reg is not None:
                    cls_loss = F.smooth_l1_loss(cls_out, confounder_reg.to(cls_out.dtype))
                else:
                    cls_loss = F.cross_entropy(cls_out, confounder_idx)
                raw_loss = reg_loss + cls_weight * cls_loss
            else:
                raw_loss = reg_loss

            # Normalize loss for gradient accumulation.
            loss = raw_loss / accum_steps

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        should_step = ((step_idx + 1) % accum_steps == 0) or ((step_idx + 1) == len(dataloader))
        if should_step:
            if scaler is not None and use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if global_step_state is not None:
                global_step_state["step"] = int(global_step_state.get("step", 0)) + 1

        total_loss += raw_loss.item()

        if max_steps is not None and (step_idx + 1) >= max_steps:
            break

    return total_loss / len(dataloader)


def resolve_micro_batch_size(model_name, model_max_len, model_batch_map, tokens_per_microbatch):
    """Choose per-model micro-batch. If not mapped, derive from token budget."""
    if model_name in model_batch_map:
        return max(1, int(model_batch_map[model_name]))
    return max(1, int(tokens_per_microbatch // max(1, model_max_len)))


def resolve_grad_accum_steps(model_max_len, micro_batch_size, target_tokens_per_step):
    """Choose grad accumulation so effective tokens/step stays roughly constant."""
    tokens_per_micro_step = max(1, model_max_len * micro_batch_size)
    return max(1, int(math.ceil(target_tokens_per_step / tokens_per_micro_step)))


def is_cuda_oom_error(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message and ("cuda" in message or "tried to allocate" in message)


def resolve_loader_settings(args, is_herbert=False, for_probe=False):
    if for_probe:
        loader_num_workers = 0
    else:
        loader_num_workers = 0 if is_herbert else max(args.num_workers, 0)
    if loader_num_workers <= 0:
        return {
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": None,
            "pin_memory": torch.cuda.is_available() and not is_herbert,
        }
    return {
        "num_workers": loader_num_workers,
        "persistent_workers": False,
        "prefetch_factor": max(args.prefetch_factor, 2 if is_herbert else 2),
        "pin_memory": torch.cuda.is_available() and not is_herbert,
    }


def build_loader(dataset, batch_size, shuffle, sampler, args, is_herbert=False, for_probe=False):
    loader_settings = resolve_loader_settings(args, is_herbert=is_herbert, for_probe=for_probe)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": loader_settings["num_workers"],
        "persistent_workers": loader_settings["persistent_workers"],
        "pin_memory": loader_settings["pin_memory"],
    }
    if loader_settings["prefetch_factor"] is not None:
        loader_kwargs["prefetch_factor"] = loader_settings["prefetch_factor"]
    return DataLoader(dataset, **loader_kwargs)


def probe_herbert_micro_batch_size(model, train_dataset, loss_fn, approach, device,
                                   args, model_max_len, n_confounder_classes,
                                   cls_weight, use_amp, sampler=None):
    candidate_start = max(1, int(getattr(args, "herbert_probe_max_batch", 6)))
    candidate_batches = list(range(candidate_start, 0, -1))
    last_error = None

    for candidate_batch in candidate_batches:
        probe_loader = build_loader(
            train_dataset, candidate_batch, shuffle=False, sampler=sampler,
            args=args, is_herbert=True, for_probe=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.0)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        try:
            train_epoch(
                model, probe_loader, optimizer, loss_fn, approach, device,
                n_confounder_classes=n_confounder_classes, cls_weight=cls_weight,
                accum_steps=1, use_amp=use_amp, scaler=scaler, scheduler=None,
                max_steps=1,
            )
            del optimizer, scaler, probe_loader
            torch.cuda.empty_cache()
            gc.collect()
            return candidate_batch
        except RuntimeError as exc:
            if not is_cuda_oom_error(exc):
                raise
            last_error = exc
            del optimizer, scaler, probe_loader
            torch.cuda.empty_cache()
            gc.collect()

    if last_error is not None:
        raise last_error
    return 1


def run_fold_with_oom_backoff(model_name, approach, loss_type, fold_idx, train_idx, val_idx,
                              texts, targets, confounders_encoded, confounder_levels,
                              tokenizer, special_token_map_encoded, args, device, is_ddp,
                              rank, world_size, local_rank, loss_fn, lr_policy, model_lr,
                              model_warmup_ratio, model_lr_floor_ratio, model_lr_schedule,
                              use_amp, n_targets, n_confounder_dims, n_confounder_classes,
                              lora_override, model_len_map, model_batch_map, skf,
                              target_cols, confounder_ohe_all, item_ids, oof_preds,
                              oof_confounders, fold_prediction_store, fold_models,
                              train_sampler, val_ohe, train_conf, val_conf, train_caps,
                              val_baseline_pred, effective_targets, st_map_for_ds,
                              max_attempts=3):
    micro_batch_size = resolve_micro_batch_size(
        model_name, get_effective_max_len(model_name, tokenizer, model_len_map, args.max_len),
        model_batch_map, args.tokens_per_microbatch,
    )
    micro_batch_size = maybe_scale_t4_micro_batch(
        device, model_name, get_effective_max_len(model_name, tokenizer, model_len_map, args.max_len),
        micro_batch_size, args,
    )
    model_max_len = get_effective_max_len(model_name, tokenizer, model_len_map, args.max_len)
    is_herbert = "herbert-large-cased" in model_name
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        train_dataset = TextRegressionDataset(
            texts, effective_targets, confounders_encoded, tokenizer,
            max_len=model_max_len,
            confounder_ohe=confounder_ohe_all if approach == "feature_concat" else None,
            approach=approach, special_token_map=st_map_for_ds,
            truncation_strategy=args.truncation_strategy,
            text_indices=train_idx,
        )
        val_dataset = TextRegressionDataset(
            texts, effective_targets, confounders_encoded, tokenizer, max_len=model_max_len,
            confounder_ohe=confounder_ohe_all if approach == "feature_concat" else None,
            approach=approach, special_token_map=st_map_for_ds,
            truncation_strategy=args.truncation_strategy,
            text_indices=val_idx,
        )

        local_train_sampler = train_sampler
        if is_ddp:
            local_train_sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=True
            )

        if is_herbert and attempt > 1:
            retry_batch = max(1, micro_batch_size // 2)
            if retry_batch >= micro_batch_size:
                retry_batch = micro_batch_size - 1
            micro_batch_size = max(1, retry_batch)
            if rank == 0:
                print(f"      [fallback] HerBERT retry {attempt}/{max_attempts} with micro_batch={micro_batch_size}")

        train_loader = build_loader(
            train_dataset, micro_batch_size, shuffle=(local_train_sampler is None),
            sampler=local_train_sampler, args=args, is_herbert=is_herbert,
            for_probe=False,
        )
        val_loader = build_loader(
            val_dataset, micro_batch_size, shuffle=False, sampler=None,
            args=args, is_herbert=is_herbert, for_probe=False,
        )

        try:
            is_herbert = "herbert-large-cased" in model_name
            model = build_confounding_model(
                model_name, approach, n_targets, n_confounder_dims,
                n_confounder_classes, args.grl_alpha, lora_override,
                device,
                stella_lora_r=args.stella_lora_r,
                stella_lora_alpha=args.stella_lora_alpha,
                stella_lora_dropout=args.stella_lora_dropout,
                stella_unfreeze_last_n_layers=args.stella_unfreeze_last_n_layers,
                stella_pooling=args.stella_pooling,
                herbert_lora_r=args.herbert_lora_r,
                herbert_lora_alpha=args.herbert_lora_alpha,
                herbert_lora_dropout=args.herbert_lora_dropout,
                herbert_unfreeze_last_n_layers=args.herbert_unfreeze_last_n_layers,
                herbert_pooling=args.herbert_pooling,
                confounder_mode=args.confounder_mode,
                stella_head_hidden_dim=args.stella_head_hidden_dim,
                stella_head_dropout=args.stella_head_dropout,
                stella_head_l2_normalize=args.stella_head_l2_normalize,
                herbert_head_hidden_dim=args.herbert_head_hidden_dim,
                herbert_head_dropout=args.herbert_head_dropout,
                herbert_head_l2_normalize=args.herbert_head_l2_normalize,
            )
            if is_ddp:
                model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                            find_unused_parameters=True)

            optimizer_kwargs = {"lr": model_lr, "weight_decay": args.weight_decay}
            if args.use_fused_adamw and torch.cuda.is_available():
                optimizer_kwargs["fused"] = True
            is_stella = is_stella_model_name(model_name)
            if is_stella and args.stella_head_lr > 0:
                use_discriminative_lr = True
                head_lr_value = args.stella_head_lr
            elif is_herbert and args.herbert_head_lr > 0:
                use_discriminative_lr = True
                head_lr_value = args.herbert_head_lr
            else:
                use_discriminative_lr = False
                head_lr_value = model_lr
            if use_discriminative_lr:
                optimizer_params = build_optimizer_param_groups(
                    model, model_lr, head_lr_value, args.weight_decay
                )
                if rank == 0:
                    print(f"      [optim] discriminative LR: encoder={model_lr:g}, "
                          f"head/neck={head_lr_value:g}")
            else:
                optimizer_params = model.parameters()
            try:
                optimizer = torch.optim.AdamW(optimizer_params, **optimizer_kwargs)
            except TypeError:
                optimizer_kwargs.pop("fused", None)
                optimizer = torch.optim.AdamW(optimizer_params, **optimizer_kwargs)
            grad_accum_steps = resolve_grad_accum_steps(
                model_max_len, micro_batch_size, args.target_tokens_per_step
            )
            total_update_steps = max(1, math.ceil(len(train_loader) / grad_accum_steps) * args.epochs)
            scheduler = None
            if lr_policy.get("schedule") == "warmup_cosine":
                scheduler = build_warmup_cosine_scheduler(
                    optimizer, total_update_steps, model_warmup_ratio, model_lr_floor_ratio
                )
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

            best_val_loss = float("inf")
            best_state = None
            patience_counter = 0

            for epoch in range(args.epochs):
                if local_train_sampler is not None:
                    local_train_sampler.set_epoch(epoch)
                train_loss = train_epoch(
                    model, train_loader, optimizer, loss_fn, approach,
                    device, n_confounder_classes, cls_weight=args.cls_weight,
                    accum_steps=grad_accum_steps, use_amp=use_amp, scaler=scaler,
                    scheduler=scheduler,
                )
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(device)
                        attn = batch["attention_mask"].to(device)
                        t = batch["targets"].to(device)
                        ohe = batch.get("confounder_ohe")
                        if ohe is not None:
                            ohe = ohe.to(device)
                        cidx = batch.get("confounder_idx")
                        if cidx is not None:
                            cidx = cidx.to(device)
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            pred, _ = model(input_ids, attn, confounder_ohe=ohe,
                                            confounder_idx=cidx)
                            val_losses.append(loss_fn(pred, t).item())
                val_loss = np.mean(val_losses)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = get_trainable_state_dict(model)
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        break

                if rank == 0 and (epoch + 1) % 5 == 0:
                    print(f"      Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if best_state is not None:
                load_trainable_state_dict(model, best_state)
                del best_state
            model.eval()

            val_preds = predict(model, val_loader, device, approach)
            if approach == "residual" and val_baseline_pred is not None:
                # Recover y by adding the ridge baseline prediction back.
                val_preds = val_preds + val_baseline_pred

            val_item_ids = [item_ids[i] for i in val_idx]
            for t_idx, tcol in enumerate(target_cols):
                fold_prediction_store[tcol].append({
                    "fold_idx": int(fold_idx),
                    "indices": np.asarray(val_idx, dtype=np.int32),
                    "item_ids": val_item_ids,
                    "y_true": np.asarray(targets[val_idx, t_idx], dtype=np.float32),
                    "y_pred": np.asarray(val_preds[:, t_idx], dtype=np.float32),
                    "confounders": np.asarray(val_conf, dtype=np.int16),
                })

            oof_preds[val_idx] = val_preds
            oof_confounders[val_idx] = val_conf

            if fold_idx == 4:
                model_for_metrics = model.module if isinstance(model, DDP) else model
                val_texts_last = [texts[i] for i in val_idx]
                # 9-tuple: see main loop for layout (target stats appended at the end).
                fold_models.append((model_for_metrics, val_idx, val_texts_last,
                                    targets[val_idx], val_conf, val_ohe, None,
                                    np.zeros(targets.shape[1], dtype=np.float32),
                                    np.ones(targets.shape[1], dtype=np.float32)))

            del train_dataset, val_dataset, train_loader, val_loader
            del model, optimizer, scaler
            torch.cuda.empty_cache()
            gc.collect()
            return micro_batch_size
        except RuntimeError as exc:
            del train_loader, val_loader
            if is_cuda_oom_error(exc) and attempt < max_attempts:
                if rank == 0:
                    print(f"      [fallback] OOM at micro_batch={micro_batch_size}; retrying smaller")
                torch.cuda.empty_cache()
                gc.collect()
                continue
            raise



def build_confounding_model(model_name, approach, n_targets, n_confounder_dims,
                            n_confounder_classes, grl_alpha, lora_target_modules,
                            device, stella_lora_r=16, stella_lora_alpha=32,
                            stella_lora_dropout=0.1, stella_unfreeze_last_n_layers=0,
                            stella_pooling="mean",
                            herbert_lora_r=16, herbert_lora_alpha=32,
                            herbert_lora_dropout=0.1, herbert_unfreeze_last_n_layers=0,
                            herbert_pooling="cls",
                            herbert_head_hidden_dim=0, herbert_head_dropout=0.0,
                            herbert_head_l2_normalize=False,
                            confounder_mode="categorical",
                            stella_head_hidden_dim=0, stella_head_dropout=0.0,
                            stella_head_l2_normalize=False,
                            feature_concat_mode="additive",
                            feature_concat_conf_hidden=64,
                            feature_concat_conf_dropout=0.2,
                            use_group_bias=True,
                            special_tokens_pool_extra=True,
                            special_tokens_pool_k=4):
    is_stella = is_stella_model_name(model_name)
    is_herbert = "herbert-large-cased" in model_name
    pooling_mode = stella_pooling if is_stella else (herbert_pooling if is_herbert else None)
    head_hidden_dim = stella_head_hidden_dim if is_stella else (herbert_head_hidden_dim if is_herbert else 0)
    head_dropout = stella_head_dropout if is_stella else (herbert_head_dropout if is_herbert else 0.0)
    head_l2_normalize = stella_head_l2_normalize if is_stella else (herbert_head_l2_normalize if is_herbert else False)
    return ConfoundingModel(
        model_name, approach, n_targets, n_confounder_dims,
        n_confounder_classes, grl_alpha=grl_alpha,
        lora_target_modules=lora_target_modules,
        stella_lora_r=stella_lora_r,
        stella_lora_alpha=stella_lora_alpha,
        stella_lora_dropout=stella_lora_dropout,
        stella_unfreeze_last_n_layers=stella_unfreeze_last_n_layers,
        herbert_lora_r=herbert_lora_r,
        herbert_lora_alpha=herbert_lora_alpha,
        herbert_lora_dropout=herbert_lora_dropout,
        herbert_unfreeze_last_n_layers=herbert_unfreeze_last_n_layers,
        pooling_mode=pooling_mode,
        confounder_mode=confounder_mode,
        head_hidden_dim=head_hidden_dim,
        head_dropout=head_dropout,
        head_l2_normalize=head_l2_normalize,
        feature_concat_mode=feature_concat_mode,
        feature_concat_conf_hidden=feature_concat_conf_hidden,
        feature_concat_conf_dropout=feature_concat_conf_dropout,
        use_group_bias=use_group_bias,
        special_tokens_pool_extra=special_tokens_pool_extra,
        special_tokens_pool_k=special_tokens_pool_k,
    ).to(device)


def build_optimizer_param_groups(model, base_lr, head_lr, weight_decay):
    """Split trainable params into encoder vs head/neck groups for discriminative LRs.

    The pooled-embedding neck and the regression/classification head are trained
    from scratch and need a much higher learning rate than the pretrained
    encoder/LoRA params. Returns AdamW param groups; the LR scheduler scales each
    group's base LR by the same multiplier, so the LR ratio is preserved.
    """
    core = model.module if isinstance(model, DDP) else model
    head_param_roots = ("head", "neck")
    head_params, encoder_params = [], []
    for name, param in core.named_parameters():
        if not param.requires_grad:
            continue
        if name.split(".")[0] in head_param_roots:
            head_params.append(param)
        else:
            encoder_params.append(param)
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": base_lr, "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    return groups


def build_param_groups_with_conf_branch(model, base_lr, head_lr, weight_decay,
                                        conf_branch_weight_decay=0.0,
                                        use_discriminative_lr=True):
    """Like build_optimizer_param_groups but isolates the conf_branch and the
    group_bias from weight decay.

    Rationale: the additive feature_concat conf_branch and the per-group bias
    embedding are the only places where group-conditional structure lives; the
    default weight decay (0.01) drives their weights toward zero, which
    reintroduces the "model ignores tag" failure seen with HerBERT. We give
    those parameters their own zero-decay group while keeping the same LR.
    """
    core = model.module if isinstance(model, DDP) else model
    head_param_roots = ("head", "neck")

    def _is_conf_branch(name):
        # AdditiveFeatureConcatHead lives at model.head.conf_branch / conf_dropout
        # and the per-group bias at model.group_bias
        return ("conf_branch" in name) or ("group_bias" in name)

    encoder_params, head_params, conf_params = [], [], []
    for name, param in core.named_parameters():
        if not param.requires_grad:
            continue
        if _is_conf_branch(name):
            conf_params.append(param)
            continue
        if name.split(".")[0] in head_param_roots:
            head_params.append(param)
        else:
            encoder_params.append(param)

    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": base_lr,
                       "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params,
                       "lr": head_lr if use_discriminative_lr else base_lr,
                       "weight_decay": weight_decay})
    if conf_params:
        groups.append({"params": conf_params,
                       "lr": head_lr if use_discriminative_lr else base_lr,
                       "weight_decay": conf_branch_weight_decay})
    return groups


def cast_trainable_params_to_fp32(model):
    """Promote trainable fp16 params to fp32 so the optimizer keeps fp32 master
    weights.

    The encoder (and its LoRA adapters + unfrozen layers) is loaded in fp16 for
    memory. AdamW then updates those fp16 weights *in place* with no fp32 master
    copy. At LRs around 5e-5 the per-step Adam update is at or below the fp16 ULP
    for typical transformer weight magnitudes (~1e-2), so the update is quantized
    away and changing the LR within that range (e.g. 5e-5 vs 8e-5) has almost no
    visible effect. Casting only the *trainable* params to fp32 (a few million
    params) fixes this with negligible memory cost; AMP autocast still runs the
    matmuls in fp16, so speed is unchanged. Frozen base weights stay fp16.

    Returns (n_promoted, n_trainable) for diagnostics.
    """
    core = model.module if isinstance(model, DDP) else model
    n_promoted = 0
    n_trainable = 0
    for param in core.parameters():
        if not param.requires_grad:
            continue
        n_trainable += 1
        if param.dtype == torch.float16:
            param.data = param.data.float()
            n_promoted += 1
    return n_promoted, n_trainable


def resolve_learning_rate_policy(model_name, args):
    """Choose a model-family-specific LR and decay policy."""
    if model_name == "sdadas/stella-pl" or model_name.endswith("/stella-pl"):
        return {
            "base_lr": args.stella_lr,
            "schedule": "warmup_cosine_late_decay",
            "warmup_ratio": args.stella_warmup_ratio,
            "lr_floor_ratio": args.stella_lr_floor_ratio,
            "decay_start_ratio": args.stella_decay_start_ratio,
        }
    if "herbert-large-cased" in model_name:
        return {
            "base_lr": args.herbert_lr,
            "schedule": args.herbert_schedule,
            "warmup_ratio": args.herbert_warmup_ratio,
            "lr_floor_ratio": args.herbert_lr_floor_ratio,
            "decay_start_ratio": args.herbert_decay_start_ratio,
        }
    if "retrieval-8k" in model_name or "polish-roberta-8k" in model_name:
        return {
            "base_lr": args.long_context_lr,
            "schedule": "warmup_cosine",
            "warmup_ratio": args.long_context_warmup_ratio,
            "lr_floor_ratio": args.long_context_lr_floor_ratio,
        }
    return {
        "base_lr": args.lr,
        "schedule": "fixed",
        "warmup_ratio": 0.0,
        "lr_floor_ratio": 1.0,
    }


def build_warmup_cosine_scheduler(optimizer, total_update_steps, warmup_ratio, min_lr_ratio):
    """Warm up linearly, then cosine decay to a non-zero floor."""
    total_update_steps = max(1, int(total_update_steps))
    warmup_steps = max(1, int(total_update_steps * warmup_ratio))
    min_lr_ratio = max(0.0, min(1.0, float(min_lr_ratio)))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        decay_steps = max(1, total_update_steps - warmup_steps)
        progress = min(1.0, float(current_step - warmup_steps + 1) / float(decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_linear_warmup_scheduler(optimizer, total_update_steps, warmup_ratio, min_lr_ratio):
    """Warm up linearly, then linearly decay to a non-zero floor."""
    total_update_steps = max(1, int(total_update_steps))
    warmup_steps = max(1, int(total_update_steps * warmup_ratio))
    min_lr_ratio = max(0.0, min(1.0, float(min_lr_ratio)))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        decay_steps = max(1, total_update_steps - warmup_steps)
        progress = min(1.0, float(current_step - warmup_steps + 1) / float(decay_steps))
        return (1.0 - progress) * (1.0 - min_lr_ratio) + min_lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_warmup_late_cosine_scheduler(optimizer, total_update_steps, warmup_ratio,
                                       min_lr_ratio, decay_start_ratio):
    """Warm up, hold high LR, then cosine decay in the final training segment."""
    total_update_steps = max(1, int(total_update_steps))
    warmup_steps = max(1, int(total_update_steps * warmup_ratio))
    min_lr_ratio = max(0.0, min(1.0, float(min_lr_ratio)))
    decay_start_ratio = max(0.0, min(1.0, float(decay_start_ratio)))
    decay_start_step = max(warmup_steps, int(total_update_steps * decay_start_ratio))

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        if current_step < decay_start_step:
            return 1.0
        decay_steps = max(1, total_update_steps - decay_start_step)
        progress = min(1.0, float(current_step - decay_start_step + 1) / float(decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def sanitize_path_component(value):
    """Make a stable filesystem-friendly name fragment."""
    return str(value).replace("/", "__").replace(" ", "_")


def prediction_bundle_path(output_dir, model_name, approach, loss_type):
    predictions_dir = os.path.join(output_dir, "predictions")
    file_name = "__".join([
        sanitize_path_component(model_name),
        sanitize_path_component(approach),
        sanitize_path_component(loss_type),
    ]) + ".pkl"
    return os.path.join(predictions_dir, file_name)


def completion_manifest_path(output_dir):
    return os.path.join(output_dir, "completed_runs.jsonl")


def combo_manifest_record(model_name, approach, loss_type, target_count):
    return {
        "model": model_name,
        "approach": approach,
        "loss": loss_type,
        "target_count": int(target_count),
    }


def load_completed_combo_keys(output_dir, csv_path, predictions_dir, target_count):
    completed = set()
    manifest_path = completion_manifest_path(output_dir)

    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if int(record.get("target_count", -1)) >= int(target_count):
                        completed.add((
                            sanitize_path_component(record.get("model", "")),
                            sanitize_path_component(record.get("approach", "")),
                            sanitize_path_component(record.get("loss", "")),
                        ))
        except Exception as exc:
            print(f"[warn] Could not read completion manifest at {manifest_path}: {exc}")

    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            if not existing_df.empty and {"model", "approach", "loss", "target"}.issubset(existing_df.columns):
                grouped = existing_df.groupby(["model", "approach", "loss"])['target'].nunique()
                for key, target_n in grouped.items():
                    if int(target_n) >= int(target_count):
                        completed.add(tuple(sanitize_path_component(part) for part in key))
        except Exception as exc:
            print(f"[warn] Could not read existing results CSV at {csv_path}: {exc}")

    if os.path.isdir(predictions_dir):
        for file_name in os.listdir(predictions_dir):
            if not file_name.endswith(".pkl"):
                continue
            bundle_path = os.path.join(predictions_dir, file_name)
            try:
                with open(bundle_path, "rb") as handle:
                    payload = pickle.load(handle)
                metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
                model_name = sanitize_path_component(metadata.get("model", ""))
                approach = sanitize_path_component(metadata.get("approach", ""))
                loss_type = sanitize_path_component(metadata.get("loss", ""))
                if model_name and approach and loss_type:
                    completed.add((model_name, approach, loss_type))
            except Exception:
                stem = file_name[:-4]
                parts = stem.split("__")
                if len(parts) < 3:
                    continue
                model_name = "__".join(parts[:-2])
                approach = parts[-2]
                loss_type = parts[-1]
                completed.add((model_name, approach, loss_type))

    return completed


def append_results_csv(csv_path, results_df):
    if results_df is None or results_df.empty:
        return
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    results_df.to_csv(csv_path, mode="a", header=not file_exists, index=False)


def append_prediction_bundle(output_dir, model_name, approach, loss_type, run_metadata,
                             oof_preds, oof_confounders, fold_prediction_store):
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    payload = {
        "metadata": run_metadata,
        "oof_preds": oof_preds,
        "oof_confounders": oof_confounders,
        "folds": fold_prediction_store,
    }
    bundle_path = prediction_bundle_path(output_dir, model_name, approach, loss_type)
    with open(bundle_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return bundle_path


def clear_run_outputs(output_dir):
    """Remove cached outputs so the next run recomputes every combination."""
    paths_to_remove = [
        os.path.join(output_dir, "results.csv"),
        os.path.join(output_dir, "results.tex"),
        completion_manifest_path(output_dir),
        os.path.join(output_dir, "ridge_baseline_results.csv"),
    ]
    for path in paths_to_remove:
        if os.path.isfile(path):
            os.remove(path)
    predictions_dir = os.path.join(output_dir, "predictions")
    if os.path.isdir(predictions_dir):
        shutil.rmtree(predictions_dir)
    os.makedirs(predictions_dir, exist_ok=True)


def append_completion_manifest(output_dir, model_name, approach, loss_type, target_count):
    manifest_path = completion_manifest_path(output_dir)
    record = combo_manifest_record(model_name, approach, loss_type, target_count)
    with open(manifest_path, "a") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return manifest_path


def clear_predictions_dir(predictions_dir):
    """Remove any stale serialized predictions from previous runs."""
    if os.path.isdir(predictions_dir):
        shutil.rmtree(predictions_dir)
    os.makedirs(predictions_dir, exist_ok=True)


def save_prediction_pickle(predictions_dir, model_name, approach, loss_type, target_col,
                           run_metadata, fold_records):
    """Serialize per-fold predictions for a single model/approach/loss/target run."""
    payload = {
        "metadata": run_metadata,
        "target": target_col,
        "folds": fold_records,
    }
    file_name = "__".join([
        sanitize_path_component(model_name),
        sanitize_path_component(approach),
        sanitize_path_component(loss_type),
        sanitize_path_component(target_col),
    ]) + ".pkl"
    file_path = os.path.join(predictions_dir, file_name)
    with open(file_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_trainable_state_dict(model):
    """Snapshot only trainable parameters to reduce host RAM pressure."""
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_trainable_state_dict(model, trainable_state):
    """Restore trainable parameter snapshot in-place."""
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, cpu_tensor in trainable_state.items():
            if name not in named_params:
                continue
            dst = named_params[name]
            dst.copy_(cpu_tensor.to(device=dst.device, dtype=dst.dtype))


def setup_distributed(args):
    """Initialize DDP if launched with torchrun and enabled via CLI.

    If the script is launched under torchrun (``WORLD_SIZE > 1``) but
    ``--enable_ddp`` was NOT passed, we auto-enable DDP and warn loudly. The
    previous behavior (return is_ddp=False, rank=0 to every process) silently
    ran the entire pipeline twice in parallel: 2x wall-clock and duplicated
    rank-gated prints. This guard prevents that footgun.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    launched_via_torchrun = world_size > 1 and "RANK" in os.environ
    if launched_via_torchrun and not args.enable_ddp and torch.cuda.is_available():
        rank_env = int(os.environ.get("RANK", "0"))
        if rank_env == 0:
            print(f"[ddp] WORLD_SIZE={world_size} detected (torchrun) but "
                  f"--enable_ddp was not passed; auto-enabling DDP to prevent "
                  f"running the pipeline {world_size}x in parallel.")
        args.enable_ddp = True
    if args.enable_ddp and world_size > 1 and torch.cuda.is_available():
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(hours=2),
        )
        device = torch.device(f"cuda:{local_rank}")
        return device, True, rank, world_size, local_rank
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, False, 0, 1, 0


def cleanup_distributed(is_ddp):
    if is_ddp and dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", "0"))])
        dist.destroy_process_group()


def maybe_scale_t4_micro_batch(device, model_name, model_max_len, micro_batch_size, args):
    """Use more VRAM on T4 when currently underutilized, mostly for shorter contexts."""
    if not args.t4_auto_scale or device.type != "cuda":
        return micro_batch_size
    dev_name = torch.cuda.get_device_name(device).lower()
    if "t4" not in dev_name:
        return micro_batch_size
    if "herbert-large-cased" in model_name:
        herbert_cap = getattr(args, "herbert_probe_max_batch", 8)
        return min(max(micro_batch_size, 6), herbert_cap)
    if model_max_len <= 1024:
        return max(micro_batch_size, args.t4_min_micro_batch_short)
    if model_max_len <= 2048:
        return max(micro_batch_size, args.t4_min_micro_batch_medium)
    return micro_batch_size


def predict(model, dataloader, device, approach):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            confounder_ohe = batch.get("confounder_ohe")
            if confounder_ohe is not None:
                confounder_ohe = confounder_ohe.to(device)
            confounder_idx = batch.get("confounder_idx")
            if confounder_idx is not None:
                confounder_idx = confounder_idx.to(device)
            # Autocast so fp32 (promoted) trainable params and fp16 frozen base
            # params coexist without dtype-mismatch errors in this no-grad path.
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                reg_out, _ = model(input_ids, attention_mask,
                                   confounder_ohe=confounder_ohe,
                                   confounder_idx=confounder_idx)
            all_preds.append(reg_out.float().cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def extract_embeddings(model, dataloader, device):
    model.eval()
    all_embs = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            embs = model.get_cls_embeddings(input_ids, attention_mask)
            all_embs.append(embs)
    return np.concatenate(all_embs, axis=0)


# ============================================================
# Special Tokens Selection
# ============================================================

def get_special_token_map(tokenizer, confounder_levels):
    """Find rare/unused tokens for each confounder level."""
    # Use tokens that are unlikely to appear in Polish text
    # We pick from high-ID single tokens that aren't common
    vocab = tokenizer.get_vocab()
    # Sort by ID descending, pick tokens that look like unused special chars
    candidates = []
    for tok, idx in sorted(vocab.items(), key=lambda x: -x[1]):
        if len(tok) <= 4 and not tok.startswith("[") and not tok.startswith("<"):
            # Check it's not a common Polish subword
            if any(c in tok for c in "§¶†‡•◊♦♣♠♥"):
                candidates.append(tok)
        if len(candidates) >= len(confounder_levels) * 2:
            break

    # Fallback: use [unusedN] style or just rare unicode
    rare_chars = ["§", "¶", "†", "‡", "◊", "♦", "♣", "♠", "♥", "♪", "♫", "⊕", "⊗", "⊘"]
    token_map = {}
    for i, level in enumerate(confounder_levels):
        if i < len(candidates):
            token_map[level] = candidates[i]
        else:
            token_map[level] = rare_chars[i % len(rare_chars)]
    return token_map


# ============================================================
# Confounding Metrics
# ============================================================

def cate_sensitivity(model, dataloader, device, confounder_levels, label_encoder,
                     tokenizer, approach, special_token_map, texts, confounders,
                     targets, max_len=512, confounder_mode="categorical",
                     confounder_level_scalar=None):
    """ATE/CATE: counterfactual predictions with swapped confounders."""
    model.eval()
    results = {}
    # Take a subset
    n_subset = min(50, len(texts))
    rng = np.random.default_rng(42)
    subset_idx = rng.choice(len(texts), n_subset, replace=False)

    def _build_feature(level_enc):
        """Build the feature_concat confounder vector for a given encoded level."""
        if confounder_mode == "ordinal":
            if confounder_level_scalar is not None:
                value = float(confounder_level_scalar[level_enc])
            else:
                value = float(level_enc)
            return np.array([value], dtype=np.float32)
        vec = np.zeros(len(confounder_levels), dtype=np.float32)
        vec[level_enc] = 1.0
        return vec

    for target_level in confounder_levels:
        target_enc = label_encoder.transform([target_level])[0]
        preds_counterfactual = []
        preds_original = []

        for i in subset_idx:
            text = str(texts[i])
            orig_conf = confounders[i]

            # Original prediction
            if approach == "special_tokens" and special_token_map:
                orig_text = f"{special_token_map.get(label_encoder.inverse_transform([orig_conf])[0], '')} {text}"
            else:
                orig_text = text

            # Counterfactual: change confounder to target_level
            if approach == "special_tokens" and special_token_map:
                cf_text = f"{special_token_map.get(target_level, '')} {text}"
            else:
                cf_text = text

            enc_orig = tokenize_with_strategy(tokenizer, orig_text, max_len, "head_tail")
            enc_cf = tokenize_with_strategy(tokenizer, cf_text, max_len, "head_tail")

            with torch.no_grad():
                input_ids = enc_orig["input_ids"].to(device)
                attn = enc_orig["attention_mask"].to(device)
                # For feature_concat, use the original confounder representation
                ohe = None
                if approach == "feature_concat":
                    ohe = torch.tensor(_build_feature(orig_conf)).unsqueeze(0).to(device)
                # Pass per-sample confounder_idx so the per-group bias (if any)
                # contributes; for CATE we want the original group's bias on the
                # original branch and the target group's bias on the CF branch.
                cidx_o = torch.tensor([int(orig_conf)], dtype=torch.long, device=device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    pred_o, _ = model(input_ids, attn, confounder_ohe=ohe,
                                      confounder_idx=cidx_o)
                preds_original.append(pred_o.float().cpu().numpy().flatten())

                input_ids_cf = enc_cf["input_ids"].to(device)
                attn_cf = enc_cf["attention_mask"].to(device)
                ohe_cf = None
                if approach == "feature_concat":
                    ohe_cf = torch.tensor(_build_feature(target_enc)).unsqueeze(0).to(device)
                cidx_cf = torch.tensor([int(target_enc)], dtype=torch.long, device=device)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    pred_cf, _ = model(input_ids_cf, attn_cf, confounder_ohe=ohe_cf,
                                       confounder_idx=cidx_cf)
                preds_counterfactual.append(pred_cf.float().cpu().numpy().flatten())

        preds_original = np.array(preds_original)
        preds_counterfactual = np.array(preds_counterfactual)
        cate = np.mean(preds_counterfactual - preds_original, axis=0)
        results[target_level] = cate

    return results


def stratified_residual_analysis(y_true, y_pred, confounders, confounder_levels, label_encoder):
    """Mean residual per confounder group."""
    residuals = y_true - y_pred
    results = {}
    for level in confounder_levels:
        enc = label_encoder.transform([level])[0]
        mask = confounders == enc
        if mask.sum() > 0:
            results[level] = {
                "mean_residual": np.mean(residuals[mask], axis=0),
                "std_residual": np.std(residuals[mask], axis=0),
                "n": int(mask.sum()),
            }
    return results


def ghost_classifier_test(embeddings, confounders, n_classes):
    """Train logistic regression on embeddings to predict confounder."""
    if len(np.unique(confounders)) < 2:
        return {"accuracy": 0.0, "chance": 1.0 / n_classes}
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, embeddings, confounders, cv=min(5, len(np.unique(confounders))),
                             scoring="accuracy")
    return {"accuracy": np.mean(scores), "chance": 1.0 / n_classes}


def distribution_shift_test(model, tokenizer, texts, targets, confounders,
                            confounder_levels, label_encoder, device, approach,
                            special_token_map, confounder_ohe_all, max_len=512,
                            batch_size=16, target_mean=None, target_std=None,
                            residual_model=None, residual_features_for_val=None):
    """Compare MAE on standard vs reversed distribution test set.

    When ``target_mean``/``target_std`` are provided (per-fold standardization),
    the model outputs are unscaled before computing MAE. For the residual
    approach, ``residual_model`` and ``residual_features_for_val`` allow adding
    back the ridge baseline; if not provided we report MAE on the residual scale
    (still useful for relative deltas).
    """
    def _postprocess(preds):
        out = preds
        if target_mean is not None and target_std is not None:
            out = (out * target_std) + target_mean
        return out
    # Create reversed distribution by resampling
    rng = np.random.default_rng(42)
    # Count per group
    counts = {}
    indices_per_group = {}
    for level in confounder_levels:
        enc = label_encoder.transform([level])[0]
        mask = confounders == enc
        indices_per_group[level] = np.where(mask)[0]
        counts[level] = mask.sum()

    # Reverse: make minority group majority and vice versa
    total = len(confounders)
    max_count = max(counts.values())
    min_count = max(min(counts.values()), 1)

    reversed_indices = []
    for level in confounder_levels:
        enc = label_encoder.transform([level])[0]
        group_idx = indices_per_group[level]
        # Reverse: give minority max_count samples, majority min_count
        target_n = max(1, total - counts[level])
        sampled = rng.choice(group_idx, size=min(target_n, len(group_idx)), replace=True)
        reversed_indices.extend(sampled)

    reversed_indices = np.array(reversed_indices)
    rng.shuffle(reversed_indices)

    # Predict on standard
    standard_dataset = TextRegressionDataset(
        texts, targets, confounders, tokenizer, max_len,
        confounder_ohe=confounder_ohe_all, approach=approach,
        special_token_map=special_token_map
    )
    standard_loader = DataLoader(standard_dataset, batch_size=batch_size, shuffle=False)
    standard_preds = predict(model, standard_loader, device, approach)
    standard_preds = _postprocess(standard_preds)
    if approach == "residual" and residual_model is not None and residual_features_for_val is not None:
        standard_preds = standard_preds + residual_model.predict(residual_features_for_val).astype(np.float32)
    standard_mae = mean_absolute_error(targets, standard_preds)

    # Predict on reversed
    rev_targets = targets[reversed_indices]
    rev_confounders = confounders[reversed_indices]
    rev_ohe = confounder_ohe_all[reversed_indices] if confounder_ohe_all is not None else None
    rev_dataset = TextRegressionDataset(
        texts, rev_targets, rev_confounders, tokenizer, max_len,
        confounder_ohe=rev_ohe, approach=approach,
        special_token_map=special_token_map,
        text_indices=reversed_indices
    )
    rev_loader = DataLoader(rev_dataset, batch_size=batch_size, shuffle=False)
    rev_preds = predict(model, rev_loader, device, approach)
    rev_preds = _postprocess(rev_preds)
    if approach == "residual" and residual_model is not None and residual_features_for_val is not None:
        rev_features = residual_features_for_val[reversed_indices] \
            if residual_features_for_val.shape[0] == len(targets) \
            else build_residual_features(rev_ohe, "categorical")
        rev_preds = rev_preds + residual_model.predict(rev_features).astype(np.float32)
    rev_mae = mean_absolute_error(rev_targets, rev_preds)

    return {"standard_mae": standard_mae, "reversed_mae": rev_mae,
            "delta_mae": abs(rev_mae - standard_mae)}


# ============================================================
# Main Training Loop
# ============================================================

def run_experiment(args):
    device, is_ddp, rank, world_size, local_rank = setup_distributed(args)
    if rank == 0:
        print(f"Device: {device}")
        if is_ddp:
            print(f"DDP enabled: world_size={world_size}, backend=nccl")

    # Load data
    df = pd.read_parquet(args.data_path)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    target_cols = [c for c in df.columns if c.endswith("WYNIK")]
    confounder_col = "GRUPA"

    # Fix empty / whitespace-only texts
    df["WPISY"] = df["WPISY"].fillna("").str.strip()
    df.loc[df["WPISY"] == "", "WPISY"] = EMPTY_TEXT_PLACEHOLDER
    n_empty = (df["WPISY"] == EMPTY_TEXT_PLACEHOLDER).sum()
    if n_empty:
        print(f"[data] Replaced {n_empty} empty WPISY rows with placeholder.")

    texts = df["WPISY"].values.tolist()
    targets = df[target_cols].values.astype(np.float32)
    confounders_raw = df[confounder_col].values
    if "ID" in df.columns:
        item_ids = df["ID"].values.tolist()
    else:
        item_ids = list(df.index.values)

    # Encode confounders
    label_encoder = LabelEncoder()
    confounders_encoded = label_encoder.fit_transform(confounders_raw)
    confounder_levels = list(label_encoder.classes_)
    n_confounder_classes = len(confounder_levels)

    # The confounder (GRUPA) reflects a quantifiable, monotonic property rather
    # than an unordered category. In ordinal mode we represent it as a single
    # standardized continuous scalar; in categorical mode we keep one-hot codes.
    confounder_scalar, confounder_level_scalar = build_confounder_scalar(
        confounder_levels, confounders_encoded
    )

    # One-hot codes (used by the categorical mode and by some diagnostics).
    ohe_encoder = OneHotEncoder(sparse_output=False)
    confounder_ohe_categorical = ohe_encoder.fit_transform(
        confounders_encoded.reshape(-1, 1)
    ).astype(np.float32)

    if args.confounder_mode == "ordinal":
        # Reuse the confounder_ohe_all variable (consumed widely below) to carry
        # the continuous scalar feature, so feature_concat and the metrics stage
        # automatically operate on the ordinal representation.
        confounder_ohe_all = confounder_scalar.reshape(-1, 1).astype(np.float32)
        confounder_reg_all = confounder_scalar.reshape(-1, 1).astype(np.float32)
    else:
        confounder_ohe_all = confounder_ohe_categorical
        confounder_reg_all = None
    n_confounder_dims = confounder_ohe_all.shape[1]
    if rank == 0:
        print(f"[confounder] mode={args.confounder_mode}, dims={n_confounder_dims}, "
              f"levels={n_confounder_classes}")
    n_targets = targets.shape[1]

    # Stratified KFold on confounder
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.resume:
        clear_run_outputs(args.output_dir)

    run_ridge_baseline_cv(
        texts,
        targets,
        confounders_encoded,
        skf,
        args.output_dir,
        target_cols,
        n_bootstrap=args.n_bootstrap,
        rank=rank,
    )

    models_list = args.models.split(",")
    model_len_map = parse_model_int_map(args.model_max_lens)
    model_batch_map = parse_model_int_map(args.model_batch_sizes)
    approaches = args.approaches.split(",")
    loss_types = args.losses.split(",")

    hf_token = os.environ.get("HF_TOKEN")
    available_models = []
    skipped_models = []
    for raw_model_name in models_list:
        model_name = resolve_model_name(raw_model_name)
        if is_model_resolvable(model_name, hf_token):
            available_models.append(model_name)
        else:
            skipped_models.append(model_name)

    if rank == 0 and skipped_models:
        print(f"[warn] Skipping unavailable models: {', '.join(skipped_models)}")

    models_list = available_models
    if not models_list:
        raise RuntimeError("No configured models could be resolved from Hugging Face.")

    predictions_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results.csv")
    completed_combo_keys = load_completed_combo_keys(
        args.output_dir, csv_path, predictions_dir, len(target_cols)
    ) if args.resume else set()

    herbert_probe_batch_cache = {}

    for model_name in models_list:
        if rank == 0:
            print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        lora_override = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
        if not lora_override:
            lora_override = None
        model_max_len = get_effective_max_len(model_name, tokenizer, model_len_map, args.max_len)
        micro_batch_size = resolve_micro_batch_size(
            model_name, model_max_len, model_batch_map, args.tokens_per_microbatch
        )
        micro_batch_size = maybe_scale_t4_micro_batch(device, model_name, model_max_len, micro_batch_size, args)
        lr_policy = resolve_learning_rate_policy(model_name, args)
        model_lr = lr_policy["base_lr"]
        model_warmup_ratio = lr_policy["warmup_ratio"]
        model_lr_floor_ratio = lr_policy["lr_floor_ratio"]
        model_lr_schedule = lr_policy["schedule"]
        use_amp = torch.cuda.is_available() and not args.no_amp

        # Prepare special token map
        special_token_map = get_special_token_map(tokenizer, confounder_levels)
        # Map from encoded int -> original level for dataset use
        special_token_map_encoded = {
            label_encoder.transform([level])[0]: special_token_map[level]
            for level in confounder_levels
        }

        for approach in approaches:
            approach = approach.strip()
            for loss_type in loss_types:
                loss_type = loss_type.strip()
                combo_key = (
                    sanitize_path_component(model_name),
                    sanitize_path_component(approach),
                    sanitize_path_component(loss_type),
                )
                if combo_key in completed_combo_keys:
                    if rank == 0:
                        print(f"\n  Approach: {approach}, Loss: {loss_type}")
                        print("    [skip] existing results/predictions found; not recomputing")
                    continue
                if rank == 0:
                    print(f"\n  Approach: {approach}, Loss: {loss_type}")

                # Storage for OOF predictions
                oof_preds = np.zeros_like(targets)
                oof_confounders = np.zeros(len(targets), dtype=int)
                fold_prediction_store = {tcol: [] for tcol in target_cols}
                fold_models = []
                combo_results = []

                # Select loss
                if loss_type == "l1":
                    loss_fn = nn.L1Loss()
                elif loss_type == "huber":
                    loss_fn = nn.SmoothL1Loss()
                else:
                    raise ValueError(f"Unknown loss: {loss_type}")

                skip_branch = False

                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(
                        np.zeros(len(targets)), confounders_encoded)):
                    if rank == 0:
                        print(f"    Fold {fold_idx + 1}/5")

                    train_conf = confounders_encoded[train_idx]
                    val_conf = confounders_encoded[val_idx]
                    val_ohe = confounder_ohe_all[val_idx]

                    # Per-fold target winsorization (train stats only, no leakage)
                    train_targets_view = targets[train_idx]
                    train_caps = None
                    if args.winsorize_pct < 100.0:
                        train_caps = np.percentile(train_targets_view, args.winsorize_pct, axis=0)
                        if rank == 0:
                            print(f"      [winsorize] caps (p{args.winsorize_pct}): "
                                f"{dict(zip(target_cols, train_caps.tolist()))}")

                    # For residual approach: train Ridge first.
                    # The deep model is trained to predict the *residual*
                    # (target - ridge.predict(confounder_features)). At inference
                    # we therefore need to add back the *baseline prediction*
                    # ridge.predict(confounder_features), NOT the held-out
                    # residual itself (that would leak the target via y_val).
                    residual_model = None
                    if approach == "residual":
                        residual_model = Ridge(alpha=1.0)
                        train_targets_clipped = np.clip(train_targets_view, None, train_caps) if train_caps is not None else train_targets_view
                        residual_features_all = build_residual_features(
                            confounder_ohe_all, args.confounder_mode
                        )
                        train_ohe = residual_features_all[train_idx]
                        residual_model.fit(train_ohe, train_targets_clipped)
                        full_targets_clipped = np.clip(targets, None, train_caps) if train_caps is not None else targets
                        full_baseline_pred = residual_model.predict(residual_features_all).astype(np.float32)
                        full_residuals = (full_targets_clipped - full_baseline_pred).astype(np.float32)
                        # Baseline ridge prediction on the validation fold,
                        # used to recover y from residual predictions below.
                        val_baseline_pred = full_baseline_pred[val_idx]
                        effective_targets = full_residuals
                    else:
                        effective_targets = targets
                        val_baseline_pred = None

                    # ============================================================
                    # Per-fold target standardization (train statistics only).
                    # Standardizing the regression target prevents the head from
                    # collapsing onto a single near-mean output under L1/Huber
                    # losses with low-capacity LoRA heads -- the root cause of
                    # the monotonic residual drift seen across approaches.
                    # Predictions are un-standardized below before the residual
                    # add-back / OOF storage.
                    # ============================================================
                    if getattr(args, "target_standardize", True):
                        train_eff = effective_targets[train_idx]
                        target_mean = train_eff.mean(axis=0).astype(np.float32)
                        target_std = train_eff.std(axis=0).astype(np.float32)
                        target_std = np.where(target_std < 1e-6, 1.0, target_std).astype(np.float32)
                        effective_targets = ((effective_targets - target_mean) / target_std).astype(np.float32)
                    else:
                        target_mean = np.zeros(effective_targets.shape[1], dtype=np.float32)
                        target_std = np.ones(effective_targets.shape[1], dtype=np.float32)

                    # ============================================================
                    # Per-group bias init from train per-group means (in the same
                    # scale the model is trained on, i.e. residuals if residual
                    # approach and/or standardized if target_standardize). This
                    # anchors each group's prediction on the correct level from
                    # epoch 1 and is the cheapest cure for the monotonic shrinkage
                    # in multitask / adversarial / special_tokens.
                    # ============================================================
                    per_group_mean_train = None
                    if getattr(args, "use_group_bias", True) and approach not in ("residual", "feature_concat"):
                        per_group_mean_train = np.zeros(
                            (n_confounder_classes, n_targets), dtype=np.float32
                        )
                        for g in range(n_confounder_classes):
                            mask = train_conf == g
                            if mask.sum() > 0:
                                per_group_mean_train[g] = effective_targets[train_idx][mask].mean(axis=0)

                    # Special token map for dataset (uses encoded ints)
                    st_map_for_ds = special_token_map_encoded if approach == "special_tokens" else None

                    # Continuous confounder target for the ordinal auxiliary task.
                    conf_reg_for_ds = (
                        confounder_reg_all
                        if (args.confounder_mode == "ordinal"
                            and approach in ("multitask", "adversarial"))
                        else None
                    )

                    train_dataset = TextRegressionDataset(
                        texts, effective_targets, confounders_encoded, tokenizer,
                        max_len=model_max_len,
                        confounder_ohe=confounder_ohe_all if approach == "feature_concat" else None,
                        approach=approach, special_token_map=st_map_for_ds,
                        truncation_strategy=args.truncation_strategy,
                        text_indices=train_idx, confounder_reg=conf_reg_for_ds
                    )
                    val_dataset = TextRegressionDataset(
                        texts, effective_targets, confounders_encoded, tokenizer, max_len=model_max_len,
                        confounder_ohe=confounder_ohe_all if approach == "feature_concat" else None,
                        approach=approach, special_token_map=st_map_for_ds,
                        truncation_strategy=args.truncation_strategy,
                        text_indices=val_idx, confounder_reg=conf_reg_for_ds
                    )

                    train_sampler = None
                    if is_ddp:
                        train_sampler = DistributedSampler(
                            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
                        )

                    is_herbert = "herbert-large-cased" in model_name
                    if is_ddp:
                        dist.barrier(device_ids=[local_rank])
                    try:
                        model = build_confounding_model(
                            model_name, approach, n_targets, n_confounder_dims,
                            n_confounder_classes, args.grl_alpha, lora_override,
                            device,
                            stella_lora_r=args.stella_lora_r,
                            stella_lora_alpha=args.stella_lora_alpha,
                            stella_lora_dropout=args.stella_lora_dropout,
                            stella_unfreeze_last_n_layers=args.stella_unfreeze_last_n_layers,
                            stella_pooling=args.stella_pooling,
                            herbert_lora_r=args.herbert_lora_r,
                            herbert_lora_alpha=args.herbert_lora_alpha,
                            herbert_lora_dropout=args.herbert_lora_dropout,
                            herbert_unfreeze_last_n_layers=args.herbert_unfreeze_last_n_layers,
                            herbert_pooling=args.herbert_pooling,
                            confounder_mode=args.confounder_mode,
                            stella_head_hidden_dim=args.stella_head_hidden_dim,
                            stella_head_dropout=args.stella_head_dropout,
                            stella_head_l2_normalize=args.stella_head_l2_normalize,
                            herbert_head_hidden_dim=args.herbert_head_hidden_dim,
                            herbert_head_dropout=args.herbert_head_dropout,
                            herbert_head_l2_normalize=args.herbert_head_l2_normalize,
                            feature_concat_mode=args.feature_concat_mode,
                            feature_concat_conf_hidden=args.feature_concat_conf_hidden,
                            feature_concat_conf_dropout=args.feature_concat_conf_dropout,
                            use_group_bias=args.use_group_bias,
                            special_tokens_pool_extra=args.special_tokens_pool_extra,
                            special_tokens_pool_k=args.special_tokens_pool_k,
                        )
                    except (RepositoryNotFoundError, OSError) as exc:
                        if rank == 0:
                            print(f"[warn] Skipping model {model_name} during load: {exc}")
                        skip_branch = True
                        break

                    # Initialize per-group bias from train per-group means so
                    # the model starts at the correct per-group level.
                    if per_group_mean_train is not None:
                        model.init_group_bias_from_means(per_group_mean_train)

                    # Promote trainable fp16 params (LoRA + unfrozen encoder
                    # layers) to fp32 master weights. Without this, AdamW updates
                    # fp16 weights in place and LR changes in the ~5e-5 range are
                    # quantized away (no visible effect from encoder LR). Applied
                    # to both Stella and HerBERT when enabled.
                    should_promote_stella = is_stella_model_name(model_name) and getattr(args, "stella_fp32_trainable", True)
                    should_promote_herbert = is_herbert and getattr(args, "herbert_fp32_trainable", True)
                    if should_promote_stella or should_promote_herbert:
                        n_promoted, n_trainable_groups = cast_trainable_params_to_fp32(model)
                        if rank == 0:
                            print(f"      [precision] promoted {n_promoted}/{n_trainable_groups} "
                                  f"trainable param tensors fp16->fp32 (master weights)")

                    if is_ddp:
                        dist.barrier(device_ids=[local_rank])

                    if is_herbert and is_ddp:
                        dist.barrier(device_ids=[local_rank])
                    if is_herbert:
                        cached_batch = herbert_probe_batch_cache.get(model_name)
                        if cached_batch is None and rank == 0:
                            probe_state = get_trainable_state_dict(model)
                            try:
                                probed_batch = probe_herbert_micro_batch_size(
                                    model, train_dataset, loss_fn, approach, device,
                                    args, model_max_len, n_confounder_classes, args.cls_weight,
                                    use_amp, sampler=train_sampler,
                                )
                                if probed_batch != micro_batch_size:
                                    print(f"      [probe] HerBERT micro-batch adjusted {micro_batch_size} -> {probed_batch}")
                                cached_batch = probed_batch
                                herbert_probe_batch_cache[model_name] = probed_batch
                            except RuntimeError as exc:
                                print(f"      [probe] HerBERT batch probing failed, keeping {micro_batch_size}: {exc}")
                            finally:
                                load_trainable_state_dict(model, probe_state)
                                del probe_state
                                torch.cuda.empty_cache()
                                gc.collect()
                        if cached_batch is not None:
                            micro_batch_size = cached_batch
                    if is_herbert and is_ddp:
                        batch_tensor = torch.tensor([micro_batch_size], device=device, dtype=torch.int64)
                        dist.broadcast(batch_tensor, src=0)
                        micro_batch_size = int(batch_tensor.item())
                        dist.barrier(device_ids=[local_rank])

                    grad_accum_steps = resolve_grad_accum_steps(
                        model_max_len, micro_batch_size, args.target_tokens_per_step
                    )
                    if rank == 0:
                        print(
                            f"  Using max_len={model_max_len}, micro_batch={micro_batch_size}, "
                            f"grad_accum={grad_accum_steps}, amp={use_amp}, lr={model_lr:g}, "
                            f"schedule={model_lr_schedule}, warmup={model_warmup_ratio:g}, "
                            f"floor={model_lr_floor_ratio:g} for model {model_name}"
                        )

                    train_loader = build_loader(
                        train_dataset, micro_batch_size, shuffle=(train_sampler is None),
                        sampler=train_sampler, args=args, is_herbert=is_herbert,
                    )
                    val_loader = build_loader(
                        val_dataset, micro_batch_size, shuffle=False, sampler=None,
                        args=args, is_herbert=is_herbert,
                    )

                    if is_ddp:
                        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                                    find_unused_parameters=True)

                    optimizer_kwargs = {"lr": model_lr, "weight_decay": args.weight_decay}
                    if args.use_fused_adamw and torch.cuda.is_available():
                        optimizer_kwargs["fused"] = True
                    is_stella = is_stella_model_name(model_name)
                    if is_stella and args.stella_head_lr > 0:
                        use_discriminative_lr = True
                        head_lr_value = args.stella_head_lr
                    elif is_herbert and args.herbert_head_lr > 0:
                        use_discriminative_lr = True
                        head_lr_value = args.herbert_head_lr
                    else:
                        use_discriminative_lr = False
                        head_lr_value = model_lr
                    # Always use the conf-branch-aware grouping when the model
                    # has either an additive conf_branch or a per-group bias;
                    # otherwise it degenerates to the original 2-group split.
                    optimizer_params = build_param_groups_with_conf_branch(
                        model, model_lr, head_lr_value, args.weight_decay,
                        conf_branch_weight_decay=args.feature_concat_conf_weight_decay,
                        use_discriminative_lr=use_discriminative_lr,
                    )
                    if rank == 0 and use_discriminative_lr:
                        group_summary = ", ".join(
                            f"lr={g['lr']:g}:{sum(p.numel() for p in g['params']):,}p"
                            for g in optimizer_params
                        )
                        print(f"      [optim] discriminative LR: encoder={model_lr:g}, "
                              f"head/neck={head_lr_value:g}, "
                              f"conf_branch_wd={args.feature_concat_conf_weight_decay:g} "
                              f"| groups[{group_summary}]")
                    try:
                        optimizer = torch.optim.AdamW(optimizer_params, **optimizer_kwargs)
                    except TypeError:
                        optimizer_kwargs.pop("fused", None)
                        optimizer = torch.optim.AdamW(optimizer_params, **optimizer_kwargs)
                    total_update_steps = max(1, math.ceil(len(train_loader) / grad_accum_steps) * args.epochs)
                    scheduler = None
                    if lr_policy.get("schedule") == "warmup_cosine":
                        scheduler = build_warmup_cosine_scheduler(
                            optimizer, total_update_steps, model_warmup_ratio, model_lr_floor_ratio
                        )
                    elif lr_policy.get("schedule") == "warmup_cosine_late_decay":
                        scheduler = build_warmup_late_cosine_scheduler(
                            optimizer,
                            total_update_steps,
                            model_warmup_ratio,
                            model_lr_floor_ratio,
                            lr_policy.get("decay_start_ratio", 0.8),
                        )
                    elif lr_policy.get("schedule") == "linear_warmup_decay":
                        scheduler = build_linear_warmup_scheduler(
                            optimizer, total_update_steps, model_warmup_ratio, model_lr_floor_ratio
                        )
                    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

                    best_val_loss = float("inf")
                    best_state = None
                    patience_counter = 0

                    # Shared step counter so the GRL alpha warmup progresses
                    # monotonically across epochs.
                    grl_step_state = {"step": 0}
                    grl_total_steps = total_update_steps

                    for epoch in range(args.epochs):
                        if train_sampler is not None:
                            train_sampler.set_epoch(epoch)
                        train_loss = train_epoch(
                            model, train_loader, optimizer, loss_fn, approach,
                            device, n_confounder_classes, cls_weight=args.cls_weight,
                            accum_steps=grad_accum_steps, use_amp=use_amp, scaler=scaler,
                            scheduler=scheduler, confounder_mode=args.confounder_mode,
                            grl_schedule=args.grl_schedule,
                            grl_alpha_max=args.grl_alpha,
                            global_step_state=grl_step_state,
                            total_train_steps=grl_total_steps,
                        )
                        # Quick val loss
                        model.eval()
                        val_losses = []
                        with torch.no_grad():
                            for batch in val_loader:
                                input_ids = batch["input_ids"].to(device)
                                attn = batch["attention_mask"].to(device)
                                t = batch["targets"].to(device)
                                ohe = batch.get("confounder_ohe")
                                if ohe is not None:
                                    ohe = ohe.to(device)
                                cidx = batch.get("confounder_idx")
                                if cidx is not None:
                                    cidx = cidx.to(device)
                                with torch.cuda.amp.autocast(enabled=use_amp):
                                    pred, _ = model(input_ids, attn,
                                                    confounder_ohe=ohe,
                                                    confounder_idx=cidx)
                                    val_losses.append(loss_fn(pred, t).item())
                        val_loss = np.mean(val_losses)

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            best_state = get_trainable_state_dict(model)
                        else:
                            patience_counter += 1
                            if patience_counter >= args.patience:
                                break

                        if rank == 0 and (epoch + 1) % 5 == 0:
                            # Report losses in raw target units so the numbers
                            # are comparable across runs with and without
                            # target standardization. For L1/Huber the loss
                            # scales linearly with target_std on average; we
                            # multiply by the mean per-target std (exact for
                            # single-target, a close approximation otherwise).
                            scale = float(np.mean(target_std)) if target_std is not None else 1.0
                            if getattr(args, "target_standardize", True) and scale != 1.0:
                                print(f"      Epoch {epoch+1}: "
                                      f"train_loss={train_loss:.4f} (raw~{train_loss*scale:.4f}), "
                                      f"val_loss={val_loss:.4f} (raw~{val_loss*scale:.4f})")
                            else:
                                print(f"      Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

                    # Load best
                    if best_state is not None:
                        load_trainable_state_dict(model, best_state)
                        del best_state
                    model.eval()

                    # Predict val
                    val_preds = predict(model, val_loader, device, approach)

                    # Undo per-fold target standardization. For the residual
                    # approach this puts predictions back on the residual scale;
                    # the ridge baseline (in y-units) is then added below.
                    val_preds = (val_preds * target_std) + target_mean

                    if approach == "residual" and val_baseline_pred is not None:
                        # Recover y from predicted residual:
                        # y_hat = f_dl(text) + ridge.predict(x_conf_val)
                        val_preds = val_preds + val_baseline_pred

                    val_item_ids = [item_ids[i] for i in val_idx]
                    for t_idx, tcol in enumerate(target_cols):
                        fold_prediction_store[tcol].append({
                            "fold_idx": int(fold_idx),
                            "indices": np.asarray(val_idx, dtype=np.int32),
                            "item_ids": val_item_ids,
                            "y_true": np.asarray(targets[val_idx, t_idx], dtype=np.float32),
                            "y_pred": np.asarray(val_preds[:, t_idx], dtype=np.float32),
                            "confounders": np.asarray(val_conf, dtype=np.int16),
                        })

                    oof_preds[val_idx] = val_preds
                    oof_confounders[val_idx] = val_conf

                    # Keep last fold model for confounding metrics
                    if fold_idx == 4:
                        model_for_metrics = model.module if isinstance(model, DDP) else model
                        val_texts_last = [texts[i] for i in val_idx]
                        # 9-tuple: (model, val_idx, val_texts, val_targets, val_conf,
                        # val_ohe, residual_model, target_mean, target_std).
                        # The last two are the per-fold target standardization
                        # statistics so CATE / distribution_shift can map model
                        # outputs back to y-units.
                        fold_models.append((model_for_metrics, val_idx, val_texts_last,
                                            targets[val_idx], val_conf, val_ohe,
                                            residual_model,
                                            target_mean.copy(), target_std.copy()))

                    # Cleanup
                    del train_dataset, val_dataset, train_loader, val_loader
                    del train_conf, val_conf, val_ohe
                    del effective_targets, val_baseline_pred
                    if approach == "residual":
                        del train_ohe
                    del residual_model
                    del model, optimizer, scaler
                    torch.cuda.empty_cache()
                    gc.collect()

                if skip_branch:
                    continue

                if is_ddp:
                    dist.barrier(device_ids=[local_rank])

                if rank == 0:
                    print(f"\n    Computing metrics for {model_name}/{approach}/{loss_type}...")
                    bootstrap_samples = args.n_bootstrap if not is_ddp else min(args.n_bootstrap, 200)

                    # Global metrics with bootstrap CI
                    metric_fns = {
                        "MAE": lambda yt, yp: mean_absolute_error(yt, yp),
                        "NP_R2": nonparametric_r2,
                        "Spearman_rho2": spearman_rho_squared,
                        "MedAE": lambda yt, yp: median_absolute_error(yt, yp),
                        "MAPE": mape,
                    }

                    for t_idx, tcol in enumerate(target_cols):
                        yt = targets[:, t_idx]
                        yp = oof_preds[:, t_idx]

                        result_row = {
                            "model": model_name, "approach": approach,
                            "loss": loss_type, "target": tcol
                        }

                        for mname, mfn in metric_fns.items():
                            mean_val, lo, hi = bootstrap_ci(yt, yp, mfn,
                                                             n_boot=bootstrap_samples)
                            interval = (hi - lo) / 2
                            result_row[f"{mname}_mean"] = mean_val
                            result_row[f"{mname}_ci"] = interval

                        # Stratified metrics
                        for level in confounder_levels:
                            enc = label_encoder.transform([level])[0]
                            mask = oof_confounders == enc
                            if mask.sum() > 0:
                                yt_s = yt[mask]
                                yp_s = yp[mask]
                                for mname, mfn in metric_fns.items():
                                    if len(yt_s) > 10:
                                        mv, lo, hi = bootstrap_ci(yt_s, yp_s, mfn,
                                                                   n_boot=bootstrap_samples)
                                        result_row[f"{mname}_{level}_mean"] = mv
                                        result_row[f"{mname}_{level}_ci"] = (hi - lo) / 2

                        # Stratified residual analysis
                        residual_info = stratified_residual_analysis(
                            yt, yp, oof_confounders, confounder_levels, label_encoder
                        )
                        for level, info in residual_info.items():
                            result_row[f"mean_residual_{level}"] = float(info["mean_residual"])

                        combo_results.append(result_row)

                    # ============ Confounding metrics (last fold) ============
                    if fold_models:
                        last_model, val_idx, val_texts_last, val_targets_last, \
                            val_conf_last, val_ohe_last, res_model, \
                            last_target_mean, last_target_std = fold_models[0]
                        last_model = last_model.to(device)

                        # Ghost classifier (embeddings extracted without special tokens)
                        val_dataset_emb = TextRegressionDataset(
                            texts, targets, confounders_encoded, tokenizer,
                            max_len=model_max_len, approach="residual",  # any approach without OHE/special tokens
                            truncation_strategy=args.truncation_strategy,
                            text_indices=val_idx)
                        emb_loader = DataLoader(val_dataset_emb, batch_size=micro_batch_size, shuffle=False)
                        embeddings = extract_embeddings(last_model, emb_loader, device)
                        ghost_result = ghost_classifier_test(embeddings, val_conf_last, n_confounder_classes)
                        print(f"    Ghost Classifier: acc={ghost_result['accuracy']:.4f}, "
                              f"chance={ghost_result['chance']:.4f}")

                        # CATE sensitivity (reported in y-units: multiply by target_std
                        # since the difference cancels target_mean).
                        cate_results = cate_sensitivity(
                            last_model, None, device, confounder_levels, label_encoder,
                            tokenizer, approach, special_token_map_encoded,
                            val_texts_last, val_conf_last, val_targets_last, max_len=model_max_len,
                            confounder_mode=args.confounder_mode,
                            confounder_level_scalar=confounder_level_scalar
                        )
                        cate_results = {
                            lvl: (np.asarray(val, dtype=np.float64) * last_target_std)
                            for lvl, val in cate_results.items()
                        }
                        print(f"    CATE shifts: {cate_results}")

                        # Distribution shift test
                        shift_result = distribution_shift_test(
                            last_model, tokenizer, val_texts_last, val_targets_last,
                            val_conf_last, confounder_levels, label_encoder, device,
                            approach, special_token_map_encoded if approach == "special_tokens" else None,
                            val_ohe_last if approach == "feature_concat" else None,
                            max_len=model_max_len, batch_size=args.batch_size,
                            target_mean=last_target_mean, target_std=last_target_std,
                            residual_model=res_model,
                            residual_features_for_val=(
                                build_residual_features(
                                    val_ohe_last, args.confounder_mode
                                ) if approach == "residual" and res_model is not None
                                else None
                            ),
                        )
                        print(f"    Distribution shift: delta_MAE={shift_result['delta_mae']:.4f}")

                        # Store confounding metrics
                        for row in combo_results:
                            if (row["model"] == model_name and row["approach"] == approach
                                    and row["loss"] == loss_type):
                                row["ghost_acc"] = ghost_result["accuracy"]
                                row["ghost_chance"] = ghost_result["chance"]
                                row["delta_mae_shift"] = shift_result["delta_mae"]
                                for level, cate_val in cate_results.items():
                                    row[f"CATE_{level}"] = float(np.mean(cate_val))

                        combo_results_df = pd.DataFrame(combo_results)
                        append_results_csv(csv_path, combo_results_df)
                        bundle_path = append_prediction_bundle(
                            args.output_dir, model_name, approach, loss_type,
                            {
                                "model": model_name,
                                "approach": approach,
                                "loss": loss_type,
                                "model_max_len": model_max_len,
                                "micro_batch_size": micro_batch_size,
                                "grad_accum_steps": grad_accum_steps,
                                "lr": model_lr,
                                "lr_schedule": model_lr_schedule,
                                "warmup_ratio": model_warmup_ratio,
                                "lr_floor_ratio": model_lr_floor_ratio,
                                "seed": args.seed,
                            },
                            oof_preds, oof_confounders, combo_results_df.to_dict(orient="records")
                        )
                        manifest_path = append_completion_manifest(
                            args.output_dir, model_name, approach, loss_type, len(target_cols)
                        )
                        generate_latex_table(combo_results_df, args.output_dir, target_cols, confounder_levels)
                        completed_combo_keys.add(combo_key)
                        if rank == 0:
                            print(f"    Saved combo outputs to {csv_path}, {bundle_path}, {manifest_path}")

                        del last_model
                        del val_dataset_emb, emb_loader, embeddings, ghost_result, cate_results, shift_result
                        torch.cuda.empty_cache()
                        gc.collect()

                if is_ddp:
                    dist.barrier(device_ids=[local_rank])

                if is_ddp and rank != 0:
                    continue

    cleanup_distributed(is_ddp)


def write_results_outputs(results_df, output_dir, target_cols, confounder_levels,
                          model_name, approach, loss_type):
    """Append CSV and LaTeX snapshots after a completed training block."""
    csv_path = os.path.join(output_dir, "results.csv")
    os.makedirs(output_dir, exist_ok=True)
    append_results_csv(csv_path, results_df)
    print(f"\nResults appended after model={model_name}, approach={approach}, loss={loss_type} to {csv_path}")

    generate_latex_table(results_df, output_dir, target_cols, confounder_levels)


def _fmt_ci(mean_val, ci_val):
    """Format a value±CI cell, or -- if NaN."""
    try:
        if np.isnan(mean_val):
            return "--"
        return f"${mean_val:.3f} \\pm {ci_val:.3f}$"
    except (TypeError, ValueError):
        return "--"


def generate_latex_table(results_df, output_dir, target_cols, confounder_levels):
    """Generate LaTeX tables (one per target) with value ± CI format.

    Tables produced:
      1. Performance metrics: MAE, NP-R², Spearman ρ², MedAE, MAPE
      2. Confounding metrics: Ghost Acc vs chance, ΔMAE (shift),
         mean residual per GRUPA level, CATE per GRUPA level
    """
    latex_path = os.path.join(output_dir, "results.tex")
    perf_metrics = ["MAE", "NP_R2", "Spearman_rho2", "MedAE", "MAPE"]

    file_exists = os.path.exists(latex_path) and os.path.getsize(latex_path) > 0
    with open(latex_path, "a") as f:
        if not file_exists:
            f.write("% Auto-generated by confounding.py\n")
            f.write("% Requires: \\usepackage{booktabs}\n\n")
        else:
            f.write("\n% Appended results block\n\n")

        for tcol in target_cols:
            subset = results_df[results_df["target"] == tcol]
            if subset.empty:
                continue

            # ---- Table 1: Performance metrics ----
            col_spec = "lll" + "c" * len(perf_metrics)
            f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
            f.write(f"\\caption{{Performance metrics — target: \\texttt{{{tcol}}}}}\n")
            f.write(f"\\label{{tab:perf:{tcol}}}\n")
            f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n")
            f.write("Model & Approach & Loss "
                    "& MAE & NP $R^2$ & Spearman $\\rho^2$ & MedAE & MAPE \\\\\n")
            f.write("\\midrule\n")

            for _, row in subset.iterrows():
                model_short = row["model"].split("/")[-1][:16]
                cells = [model_short, row["approach"], row["loss"]]
                for mname in perf_metrics:
                    cells.append(_fmt_ci(row.get(f"{mname}_mean", np.nan),
                                         row.get(f"{mname}_ci", np.nan)))
                f.write(" & ".join(cells) + " \\\\\n")

            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

            # ---- Table 2: Confounding metrics ----
            ghost_chance = 1.0 / max(len(confounder_levels), 1)
            n_conf_cols = 2 + len(confounder_levels) * 2  # ghost+delta + residual+CATE per level
            col_spec2 = "lll" + "c" * n_conf_cols
            f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
            f.write(f"\\caption{{Confounding metrics — target: \\texttt{{{tcol}}}}}\n")
            f.write(f"\\label{{tab:conf:{tcol}}}\n")
            f.write(f"\\begin{{tabular}}{{{col_spec2}}}\n\\toprule\n")

            # Header row
            level_headers = " & ".join(
                [f"Res.~{lv} & CATE~{lv}" for lv in confounder_levels]
            )
            f.write(f"Model & Approach & Loss "
                    f"& Ghost~Acc (chance={ghost_chance:.2f}) & $\\Delta$MAE "
                    f"& {level_headers} \\\\\n")
            f.write("\\midrule\n")

            for _, row in subset.iterrows():
                model_short = row["model"].split("/")[-1][:16]
                cells = [model_short, row["approach"], row["loss"]]

                ghost = row.get("ghost_acc", np.nan)
                delta = row.get("delta_mae_shift", np.nan)
                cells.append(f"${ghost:.3f}$" if not np.isnan(float(ghost)) else "--")
                cells.append(f"${delta:.3f}$" if not np.isnan(float(delta)) else "--")

                for lv in confounder_levels:
                    res_val = row.get(f"mean_residual_{lv}", np.nan)
                    cate_val = row.get(f"CATE_{lv}", np.nan)
                    try:
                        cells.append(f"${float(res_val):.3f}$" if not np.isnan(float(res_val)) else "--")
                    except (TypeError, ValueError):
                        cells.append("--")
                    try:
                        cells.append(f"${float(cate_val):.3f}$" if not np.isnan(float(cate_val)) else "--")
                    except (TypeError, ValueError):
                        cells.append("--")

                f.write(" & ".join(cells) + " \\\\\n")

            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    print(f"LaTeX tables saved to {latex_path}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Confounding-aware regression with Transformers")
    parser.add_argument("--data_path", type=str, default="df_data.parquet",
                        help="Path to parquet data file")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing CSV/prediction outputs instead of recomputing all combinations")
    parser.add_argument("--models", type=str,
                        default=(
                            "sdadas/stella-pl,allegro/herbert-large-cased,"
                            "sdadas/stella-pl-retrieval-8k,sdadas/polish-roberta-8k"
                        ),
                        help="Comma-separated HuggingFace model names")
    parser.add_argument("--model_max_lens", "--model-max-lens", dest="model_max_lens", type=str,
                        default=(
                            "sdadas/stella-pl=512,allegro/herbert-large-cased=512,"
                            "sdadas/stella-pl-retrieval-8k=8192,sdadas/polish-roberta-8k=8192,"
                            "stella-pl-retrieval-8k=8192,polish-roberta-8k=8192"
                        ),
                        help=(
                            "Model-specific max lengths as 'model=len,model=len'. "
                            "Short aliases stella-pl-retrieval-8k and polish-roberta-8k are supported."
                        ))
    parser.add_argument("--approaches", type=str,
                        default="feature_concat,multitask,adversarial,residual,special_tokens",
                        help="Comma-separated approaches")
    parser.add_argument("--losses", type=str, default="l1,huber",
                        help="Comma-separated loss types (l1, huber)")
    parser.add_argument("--max_len", type=int, default=512,
                        help="Max token length")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Deprecated fallback batch size; overridden by model-aware micro-batching")
    parser.add_argument("--model_batch_sizes", "--model-batch-sizes", dest="model_batch_sizes", type=str,
                        default=(
                            "sdadas/stella-pl=8,allegro/herbert-large-cased=4,"
                            "sdadas/stella-pl-retrieval-8k=1,sdadas/polish-roberta-8k=1,"
                            "stella-pl-retrieval-8k=1,polish-roberta-8k=1"
                        ),
                        help=(
                            "Model-specific micro-batch sizes as 'model=bs,model=bs'. "
                            "Used to fit T4 memory limits."
                        ))
    parser.add_argument("--tokens_per_microbatch", "--tokens-per-microbatch",
                        dest="tokens_per_microbatch", type=int, default=4096,
                        help="If model batch size is not specified, derive micro-batch as floor(tokens_per_microbatch/max_len)")
    parser.add_argument("--target_tokens_per_step", "--target-tokens-per-step",
                        dest="target_tokens_per_step", type=int, default=4096,
                        help="Set grad accumulation so effective tokens per optimizer step ~= this value")
    parser.add_argument("--num_workers", "--num-workers", dest="num_workers", type=int, default=0,
                        help="Dataloader workers")
    parser.add_argument("--prefetch_factor", "--prefetch-factor", dest="prefetch_factor", type=int, default=2,
                        help="Dataloader prefetch factor (used only when num_workers > 0)")
    parser.add_argument("--herbert_probe_max_batch", "--herbert-probe-max-batch",
                        dest="herbert_probe_max_batch", type=int, default=8,
                        help="Upper bound for automatic HerBERT batch probing on T4")
    parser.add_argument("--no_amp", "--no-amp", dest="no_amp", action="store_true",
                        help="Disable fp16 mixed precision (AMP).")
    parser.add_argument("--enable_ddp", "--enable-ddp", dest="enable_ddp", action="store_true",
                        help="Enable DistributedDataParallel when launched with torchrun (WORLD_SIZE>1)")
    parser.add_argument("--use_fused_adamw", "--use-fused-adamw", dest="use_fused_adamw",
                        action="store_true",
                        help="Use fused AdamW optimizer on CUDA when available")
    parser.add_argument("--t4_auto_scale", "--t4-auto-scale", dest="t4_auto_scale", action="store_true",
                        help="Auto-increase micro-batch on NVIDIA T4 for better VRAM utilization")
    parser.add_argument("--t4_min_micro_batch_short", "--t4-min-micro-batch-short",
                        dest="t4_min_micro_batch_short", type=int, default=16,
                        help="Minimum micro-batch for T4 when max_len <= 1024")
    parser.add_argument("--t4_min_micro_batch_medium", "--t4-min-micro-batch-medium",
                        dest="t4_min_micro_batch_medium", type=int, default=4,
                        help="Minimum micro-batch for T4 when 1024 < max_len <= 2048")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Max training epochs per fold")
    parser.add_argument("--lr", type=float, default=1.5e-5,
                        help="Learning rate")
    parser.add_argument("--stella_lr", type=float, default=5e-5,
                        help="Peak learning rate for sdadas/stella-pl")
    parser.add_argument("--stella_fp32_trainable", "--stella-fp32-trainable",
                        dest="stella_fp32_trainable",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Keep fp32 master weights for Stella's trainable params "
                             "(LoRA + unfrozen layers). Required for --stella_lr to have a "
                             "measurable effect: with fp16 master weights, Adam updates at "
                             "~5e-5 are quantized to the fp16 ULP and the LR is effectively "
                             "ignored. Disable to restore the old in-place fp16 behavior.")
    parser.add_argument("--stella_lora_r", type=int, default=32,
                        help="LoRA rank for sdadas/stella-pl")
    parser.add_argument("--stella_lora_alpha", type=int, default=64,
                        help="LoRA alpha for sdadas/stella-pl")
    parser.add_argument("--stella_lora_dropout", type=float, default=0.05,
                        help="LoRA dropout for sdadas/stella-pl")
    parser.add_argument("--stella_unfreeze_last_n_layers", type=int, default=4,
                        help="Number of final encoder layers to unfreeze for sdadas/stella-pl "
                             "(was 2; increased to 4 because at 2 the encoder leaves the "
                             "embedding nearly group-blind and forces the head to over-rely "
                             "on the OHE/tag.)")
    parser.add_argument("--stella_head_hidden_dim", type=int, default=1024,
                        help="Hidden width of the Stella projection neck (LayerNorm+Linear+GELU+Dropout) "
                             "applied before the task head; 0 disables it (single linear head)")
    parser.add_argument("--stella_head_dropout", type=float, default=0.05,
                        help="Dropout inside the Stella projection neck")
    parser.add_argument("--stella_head_lr", type=float, default=1e-3,
                        help="Learning rate for the Stella head+neck params (discriminative LR). "
                             "0 disables the separate group and uses a single LR for all params.")
    parser.add_argument("--stella_head_l2_normalize", action="store_true",
                        help="L2-normalize the pooled Stella embedding (after the neck) before the head")
    parser.add_argument("--stella_pooling", type=str, default="mean",
                        choices=["mean", "cls"],
                        help="Pooling for sdadas/stella-pl (mean recommended for embedding models)")
    parser.add_argument("--herbert_lr", type=float, default=5e-5,
                        help="Default learning rate for allegro/herbert-large-cased")
    parser.add_argument("--herbert_lora_r", type=int, default=32,
                        help="LoRA rank for allegro/herbert-large-cased")
    parser.add_argument("--herbert_lora_alpha", type=int, default=64,
                        help="LoRA alpha for allegro/herbert-large-cased")
    parser.add_argument("--herbert_lora_dropout", type=float, default=0.05,
                        help="LoRA dropout for allegro/herbert-large-cased")
    parser.add_argument("--herbert_unfreeze_last_n_layers", type=int, default=4,
                        help="Number of final encoder layers to unfreeze for allegro/herbert-large-cased")
    parser.add_argument("--herbert_head_hidden_dim", type=int, default=1024,
                        help="Hidden width of the HerBERT projection neck (LayerNorm+Linear+GELU+Dropout) "
                             "applied before the task head; 0 disables it (single linear head)")
    parser.add_argument("--herbert_head_dropout", type=float, default=0.05,
                        help="Dropout inside the HerBERT projection neck")
    parser.add_argument("--herbert_head_lr", type=float, default=1e-3,
                        help="Learning rate for the HerBERT head+neck params (discriminative LR). "
                             "0 disables the separate group and uses a single LR for all params.")
    parser.add_argument("--herbert_head_l2_normalize", action="store_true",
                        help="L2-normalize the pooled HerBERT embedding (after the neck) before the head")
    parser.add_argument("--herbert_pooling", type=str, default="cls",
                        choices=["mean", "cls"],
                        help="Pooling for allegro/herbert-large-cased (cls recommended for encoder models)")
    parser.add_argument("--herbert_fp32_trainable", "--herbert-fp32-trainable",
                        dest="herbert_fp32_trainable",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Keep fp32 master weights for HerBERT's trainable params "
                             "(LoRA + unfrozen layers). Required for --herbert_lr to have a "
                             "measurable effect: with fp16 master weights, Adam updates at "
                             "~5e-5 are quantized to the fp16 ULP and the LR is effectively "
                             "ignored. Disable to restore the old in-place fp16 behavior.")
    parser.add_argument("--long_context_lr", type=float, default=1.0e-5,
                        help="Default learning rate for long-context 8k models")
    parser.add_argument("--stella_warmup_ratio", type=float, default=0.1,
                        help="Warmup fraction for sdadas/stella-pl")
    parser.add_argument("--stella_decay_start_ratio", type=float, default=0.8,
                        help="Step ratio at which sdadas/stella-pl begins cosine decay")
    parser.add_argument("--stella_lr_floor_ratio", type=float, default=0.0,
                        help="Minimum LR fraction for sdadas/stella-pl")
    parser.add_argument("--herbert_warmup_ratio", type=float, default=0.1,
                        help="Warmup fraction for allegro/herbert-large-cased")
    parser.add_argument("--herbert_lr_floor_ratio", type=float, default=0.0,
                        help="Minimum LR fraction for allegro/herbert-large-cased")
    parser.add_argument("--herbert_schedule", type=str, default="warmup_cosine_late_decay",
                        choices=["fixed", "warmup_cosine", "warmup_cosine_late_decay",
                                 "linear_warmup_decay"],
                        help="LR schedule for allegro/herbert-large-cased")
    parser.add_argument("--herbert_decay_start_ratio", type=float, default=0.8,
                        help="Step ratio at which HerBERT begins cosine decay "
                             "(used by warmup_cosine_late_decay)")
    parser.add_argument("--long_context_warmup_ratio", type=float, default=0.1,
                        help="Warmup fraction for long-context 8k models")
    parser.add_argument("--long_context_lr_floor_ratio", type=float, default=0.2,
                        help="Minimum LR fraction for long-context 8k models")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--winsorize_pct", type=float, default=99.0,
                        help="Per-fold winsorization percentile for targets (100=disabled). "
                             "Applied to training set only; val targets are clipped to the same cap.")
    parser.add_argument("--truncation_strategy", type=str, default="head_tail",
                        choices=["head", "tail", "head_tail"],
                        help="Tokenization truncation: head=first N tokens, "
                             "tail=last N tokens, head_tail=first 128 + last 382 tokens")
    parser.add_argument("--grl_alpha", type=float, default=1.0,
                        help="GRL alpha for adversarial approach")
    parser.add_argument("--confounder_mode", "--confounder-mode", dest="confounder_mode",
                        type=str, default="ordinal", choices=["ordinal", "categorical"],
                        help=(
                            "How to treat the confounder (GRUPA). 'ordinal' treats it as a "
                            "standardized continuous, monotonic quantity (scalar feature, "
                            "regression aux task, polynomial residual model); 'categorical' "
                            "uses one-hot codes and classification."
                        ))
    parser.add_argument("--lora_target_modules", "--lora-target-modules",
                        dest="lora_target_modules", type=str, default="",
                        help=(
                            "Optional comma-separated LoRA target module names override. "
                            "If empty, modules are inferred from model architecture."
                        ))
    parser.add_argument("--cls_weight", type=float, default=0.5,
                        help="Classification loss weight for multitask/adversarial")
    # ------------------------------------------------------------------
    # Confounding-method-specific knobs (new in this revision).
    # All defaults are chosen to remove the monotonic residual drift and
    # ignore-the-tag / over-rely-on-the-tag failures seen in the v1 results.
    # ------------------------------------------------------------------
    parser.add_argument("--feature_concat_mode", "--feature-concat-mode",
                        dest="feature_concat_mode", type=str, default="additive",
                        choices=["additive", "concat"],
                        help=(
                            "feature_concat head architecture. 'additive' uses two "
                            "decoupled branches y = text(cls) + conf_mlp(ohe) with per-sample "
                            "dropout on the OHE input (fixes both HerBERT 'ignores tag' and "
                            "Stella 'over-relies on tag'). 'concat' is the legacy "
                            "Linear([cls||ohe])."
                        ))
    parser.add_argument("--feature_concat_conf_hidden", "--feature-concat-conf-hidden",
                        dest="feature_concat_conf_hidden", type=int, default=64,
                        help="Hidden width of the additive feature_concat conf branch MLP.")
    parser.add_argument("--feature_concat_conf_dropout", "--feature-concat-conf-dropout",
                        dest="feature_concat_conf_dropout", type=float, default=0.2,
                        help=(
                            "Per-sample dropout on the OHE input of the additive "
                            "feature_concat head. >0 forces the text branch to carry signal "
                            "even when the tag is masked."
                        ))
    parser.add_argument("--feature_concat_conf_weight_decay",
                        "--feature-concat-conf-weight-decay",
                        dest="feature_concat_conf_weight_decay", type=float, default=0.0,
                        help=(
                            "Weight decay applied to the additive feature_concat conf branch "
                            "and to the per-group bias. Default 0.0 prevents the global "
                            "--weight_decay from crushing the only group-conditional weights "
                            "in the head."
                        ))
    parser.add_argument("--use_group_bias", "--use-group-bias",
                        dest="use_group_bias", action=argparse.BooleanOptionalAction,
                        default=True,
                        help=(
                            "Add a learnable per-group bias Embedding(n_classes, n_targets) "
                            "on top of the regression head, initialized from per-fold train "
                            "per-group means. Active for multitask/adversarial/special_tokens. "
                            "Use --no-use_group_bias to disable."
                        ))
    parser.add_argument("--target_standardize", "--target-standardize",
                        dest="target_standardize", action=argparse.BooleanOptionalAction,
                        default=True,
                        help=(
                            "Z-score the regression target per-fold with train statistics. "
                            "Strongly recommended under L1/Huber on low-capacity LoRA heads."
                        ))
    parser.add_argument("--special_tokens_pool_extra", "--special-tokens-pool-extra",
                        dest="special_tokens_pool_extra", action=argparse.BooleanOptionalAction,
                        default=True,
                        help=(
                            "For approach=special_tokens: concatenate the mean hidden state "
                            "of the first K post-CLS positions (where the prepended group "
                            "marker lands) with the standard pooled embedding before the head."
                        ))
    parser.add_argument("--special_tokens_pool_k", "--special-tokens-pool-k",
                        dest="special_tokens_pool_k", type=int, default=4,
                        help="K for --special_tokens_pool_extra (default 4 post-CLS positions).")
    parser.add_argument("--grl_schedule", "--grl-schedule",
                        dest="grl_schedule", type=str, default="gan",
                        choices=["fixed", "gan"],
                        help=(
                            "GRL alpha schedule for the adversarial approach. 'gan' uses the "
                            "Ganin-Lempitsky warmup alpha(p) = grl_alpha * (2/(1+exp(-10p))-1) "
                            "with p = global_step / total_train_steps. 'fixed' keeps "
                            "alpha = --grl_alpha for the entire run."
                        ))
    parser.add_argument("--n_bootstrap", type=int, default=1000,
                        help="Number of bootstrap samples for CI")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[warn] Ignoring unknown CLI arguments: {' '.join(unknown)}")
    return args


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    run_experiment(args)
