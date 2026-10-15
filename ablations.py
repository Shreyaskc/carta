"""Ablation registry (pure Python - no jax/flax, so it reads in any env).

Each ablation is a set of overrides applied on top of the full CARTA config.
Two kinds, honestly labeled:

  RUNNABLE_NOW  - pure config/flag toggles, run today on the cluster with existing
                 code + encoders.py (no further implementation needed).
  NEEDS_CODE    - require a code variant; the hook exists, but the variant must be
                 implemented before the row is trustworthy. Marked `needs_code`.

Maps to the paper's C1 (encoder) and C2 (excision) studies plus the training-element ablations
that double as core-component evidence (token excision, contrastive/InfoNCE, cyclical KL,
GRU vs Perceiver, multi-head decoder).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Ablation:
    name: str
    claim: str
    encoder: str = "perceiver"              # perceiver | gru | mlp
    use_contrastive: bool = True
    excision_mode: str = "cell"             # cell (Table 3 reference) | feature (main default)
    model: Dict = field(default_factory=dict)     # ModelConfig overrides
    train: Dict = field(default_factory=dict)     # TrainingConfig overrides
    needs_code: str = ""                    # non-empty => not yet trustworthy
    note: str = ""


# --- RUNNABLE NOW (config/flag only) ------------------------------------------
RUNNABLE: List[Ablation] = [
    Ablation("full", "reference",
             note="Perceiver + token excision + InfoNCE + cyclical KL"),
    Ablation("no_contrastive", "InfoNCE contrastive objective", use_contrastive=False,
             train={"augmentation_mask_prob": 0.0},
             note="compound ablation: no contrastive loss and no excision"),
    Ablation("no_excision", "token excision",
             train={"augmentation_mask_prob": 0.0},
             note="contrastive kept but views identical -> excision's contribution"),
    # C2 headline: whole-feature (token) excision vs the default cell-wise corruption.
    # 'full' above is cell-mode (default) -> compare against this row.
    Ablation("excision_feature", "C2 (whole-feature vs cell excision)",
             excision_mode="feature",
             note="whole-feature token excision: drop whole feature (cont col / "
                  "one-hot block / embedding->unknown). vs 'full' (cell). vs SubTab/SCARF."),
    Ablation("no_kl_anneal", "cyclical KL annealing",
             train={"use_cyclical_annealing": False},
             note="fixed KL beta=1; annealing disabled, KL retained"),
    # SCARF comparison: same architecture, marginal-shuffle corruption instead of excision
    Ablation("scarf_augment", "SCARF vs token excision",
             excision_mode="scarf",
             note="SCARF-style marginal shuffling: strongest comparison for token excision"),
    Ablation("gru_encoder", "GRU-gated encoder variant", encoder="gru",
             note="single context vector + update/reset gates"),
    Ablation("mlp_encoder", "C1 / perceiver value", encoder="mlp",
             note="no attention bottleneck control"),
    # capacity sweeps
    Ablation("z8",  "latent-dim sweep", model={"latent_dim": 8}),
    Ablation("z32", "latent-dim sweep", model={"latent_dim": 32}),
    Ablation("z64", "latent-dim sweep", model={"latent_dim": 64}),
    Ablation("q1",  "num-queries sweep", model={"num_latent_queries": 1}),
    # q16: double the latent array -> needs larger GPU allocation; halve batch size to stay
    # within the available device memory; actual requirements must be measured.
    Ablation("q16", "num-queries sweep", model={"num_latent_queries": 16},
             train={"batch_size": 8192},
             note="halved batch_size guards against OOM on virtualized GPU slices"),
]

# --- NEEDS CODE (variant not yet implemented; do NOT report as final) --------
NEEDS_CODE: List[Ablation] = [
    Ablation("single_head_decoder", "single-head vs multi-head decoder",
             needs_code="Add a single-shared-head decoder variant to contrast with the "
                        "the type-specific multi-head Decoder.",
             note="does type-specific decoding matter?"),
]

ALL = {a.name: a for a in RUNNABLE + NEEDS_CODE}


def runnable_names() -> List[str]:
    return [a.name for a in RUNNABLE]
