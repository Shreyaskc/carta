"""CPU feature/probe demonstration. No dataset downloads or CARTA encoder training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from adapter import split_roles
from baselines import build_freq_matrix
from datasets import DatasetSpec
from probe import split


def main():
    index = np.arange(120)
    frame = pd.DataFrame({"amount": index.astype(float),
                          "category": np.where(index % 2, "a", "b"),
                          "merchant": [f"m{i % 20}" for i in index]})
    y = (index % 7 == 0).astype(int)
    spec = DatasetSpec("synthetic", ["amount"], ["category", "merchant"], "fraud")
    roles = split_roles(frame, spec, card_threshold=5)
    features = build_freq_matrix(frame, spec, card_threshold=5)
    train, test = split(y, seed=7)
    classifier = LogisticRegression(max_iter=200, class_weight="balanced")
    classifier.fit(features[train], y[train])
    probability = classifier.predict_proba(features[test])[:, 1]
    assert np.isfinite(probability).all()
    print("Synthetic feature/probe example; not a CARTA embedding or benchmark result.")
    print(roles.summary())
    print(f"features={features.shape}; train={len(train)}; test={len(test)}")
    print("finite held-out probabilities: yes")


if __name__ == "__main__":
    main()
