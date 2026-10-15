"""Adapter: DatasetSpec -> CARTA feature roles (continuous / one-hot / embedding).

This is the operational form of the reference design's low/high-cardinality boundary, which
the [45]/[56] revisions clarified as a *configurable functional threshold* rather
than a fixed number. Here it is one knob: `card_threshold`.

  - numeric columns            -> continuous
  - categorical, nunique <= T  -> one_hot   (low cardinality)
  - categorical, nunique  > T  -> embedding (high cardinality)

Default T=50 matches the review-note guidance ("on the order of a few dozen").
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import pandas as pd
from datasets import DatasetSpec


@dataclass
class GaterFeatureRoles:
    continuous: List[str]
    one_hot: List[str]
    embedding: List[str]

    def summary(self) -> str:
        return (f"continuous={len(self.continuous)} one_hot={len(self.one_hot)} "
                f"embedding={len(self.embedding)}")


def split_roles(df: pd.DataFrame, spec: DatasetSpec,
                card_threshold: int = 50) -> GaterFeatureRoles:
    one_hot, embedding = [], []
    for c in spec.categorical:
        (one_hot if df[c].nunique() <= card_threshold else embedding).append(c)
    return GaterFeatureRoles(continuous=list(spec.numeric),
                             one_hot=one_hot, embedding=embedding)
