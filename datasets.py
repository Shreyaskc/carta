"""Public, double-blind-safe transaction datasets for the GATER ICAIF study.

Each loader returns ``(df, DatasetSpec)`` where DatasetSpec declares the numeric /
categorical columns and the binary fraud label. No dataset bytes are distributed with this package.

Datasets
--------
- IEEE-CIS fraud   : local parquet, raw features
                    only (engineered/leaky columns excluded). ~590K rows, 3.5% fraud.
- ULB credit-card  : OpenML data_id=1597 (needs network on first fetch, then cached
                    by sklearn).  284,807 rows, 0.173% fraud, PCA features V1..V28.

The retained IEEE feature list is declared explicitly below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple
import pandas as pd


@dataclass
class DatasetSpec:
    """Declares column roles for a dataset. `label` is the binary fraud target."""
    name: str
    numeric: List[str]
    categorical: List[str]          # ALL categoricals (adapter splits low/high card)
    label: str
    n_rows: int = 0
    fraud_rate: float = 0.0
    card: Dict[str, int] = field(default_factory=dict)  # per-categorical cardinality
    row_fingerprint: str | None = None
    encoding_protocol: Dict = field(default_factory=dict)
    time_col: str | None = None     # temporal-split key for drift (NOT a feature)

    @property
    def features(self) -> List[str]:
        return self.numeric + self.categorical


# --- IEEE-CIS: repository-local parquet ----------------
_LOCAL_IEEE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           ".data", "ieee.parquet")
_IEEE_URL = ("https://huggingface.co/datasets/Kshitijbhatt1998/"
             "ieee-fraud-detection-pipeline-features/resolve/main/fraud_features.parquet")

# Raw attributes only. EXCLUDE engineered/target-leaking columns
_IEEE_CAT = ["product_cd", "card4", "card6"]
_IEEE_NUM = ["transaction_amt", "card1", "card2", "card3", "card5", "addr1", "addr2",
             "C1", "C2", "C5", "C13", "C14", "D1", "D10", "D15"]


def _ieee_path() -> str:
    for p in (_LOCAL_IEEE,):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"IEEE parquet not found at {_LOCAL_IEEE}.\n"
        f"Download once with:  curl -sL -o {_LOCAL_IEEE} '{_IEEE_URL}'")


def load_ieee_cis() -> Tuple[pd.DataFrame, DatasetSpec]:
    raw = pd.read_parquet(_ieee_path())
    df = pd.DataFrame()
    for c in _IEEE_CAT:
        df[c] = raw[c].astype(object).where(raw[c].notna(), "missing").astype(str)
    for c in _IEEE_NUM:
        col = pd.to_numeric(raw[c], errors="coerce")
        df[c] = col.fillna(col.median())
    df["is_fraud"] = raw["is_fraud"].astype(int)
    card = {c: int(df[c].nunique()) for c in _IEEE_CAT}
    spec = DatasetSpec("ieee_cis", _IEEE_NUM, _IEEE_CAT, "is_fraud",
                       n_rows=len(df), fraud_rate=float(df["is_fraud"].mean()), card=card)
    return df, spec


# --- IEEE-CIS high-cardinality variant (review #5: real-data high-card validation) ---
# Treats card1 (13,553 unique), the email-domain strings, and the ID-like addr/card
# codes as CATEGORICALS (embedded when card > threshold) instead of z-scored numerics,
# so the high-cardinality claim is validated on REAL data (card1 cannot be one-hot
# encoded at 590K rows), not only on synthetic TabFormer. Engineered/leaky columns
# (card1_historical_fraud_rate, *_txn_count, amt_vs_card_avg_ratio, ...) are excluded.
# NOTE: DeviceInfo is NOT present in this parquet, so it is not used.
_IEEE_HC_NUM = ["transaction_amt", "C1", "C2", "C5", "C13", "C14", "D1", "D10", "D15"]
_IEEE_HC_CAT = ["product_cd", "card4", "card6",                      # low-card -> one-hot
                "card1", "card2", "addr1",                          # ID-like -> high-card embed
                "purchaser_email_domain", "recipient_email_domain"]  # string domains


def load_ieee_cis_highcard() -> Tuple[pd.DataFrame, DatasetSpec]:
    raw = pd.read_parquet(_ieee_path())
    missing = [c for c in _IEEE_HC_CAT + _IEEE_HC_NUM if c not in raw.columns]
    if missing:
        raise KeyError(f"ieee_cis_hc: columns absent from parquet: {missing}")
    df = pd.DataFrame()
    for c in _IEEE_HC_CAT:
        df[c] = raw[c].astype(object).where(raw[c].notna(), "missing").astype(str)
    for c in _IEEE_HC_NUM:
        col = pd.to_numeric(raw[c], errors="coerce")
        df[c] = col.fillna(col.median())
    df["is_fraud"] = raw["is_fraud"].astype(int)
    card = {c: int(df[c].nunique()) for c in _IEEE_HC_CAT}
    spec = DatasetSpec("ieee_cis_hc", _IEEE_HC_NUM, _IEEE_HC_CAT, "is_fraud",
                       n_rows=len(df), fraud_rate=float(df["is_fraud"].mean()), card=card)
    return df, spec


_ULB_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          ".data", "ulb_creditcard.parquet")

def load_creditcard() -> Tuple[pd.DataFrame, DatasetSpec]:
    # Try local parquet first (no internet needed on air-gapped clusters)
    if os.path.exists(_ULB_LOCAL):
        frame = pd.read_parquet(_ULB_LOCAL).copy()
    else:
        from sklearn.datasets import fetch_openml
        frame = fetch_openml(data_id=1597, as_frame=True).frame.copy()
    if "Class" in frame.columns:
        labels = frame["Class"].astype(str).str.strip("'").astype(int)
        if "is_fraud" in frame.columns and not labels.equals(frame["is_fraud"].astype(int)):
            raise ValueError("Conflicting Class and is_fraud labels")
        frame["is_fraud"] = labels
        frame = frame.drop(columns=["Class"])
    if "is_fraud" not in frame.columns:
        raise ValueError("ULB data requires Class or is_fraud labels")
    numeric = [c for c in frame.columns if c != "is_fraud"]
    for c in numeric:
        frame[c] = frame[c].astype(float)
    spec = DatasetSpec("ulb_creditcard", numeric, [], "is_fraud",
                       n_rows=len(frame), fraud_rate=float(frame["is_fraud"].mean()))
    return frame, spec


# --- BAF: Bank Account Fraud (Feedzai, NeurIPS 2022) --------------------------
# 1M rows, ~1.1% fraud (Base). Has a `month` column (0-7) for temporal drift and
# protected attributes for fairness. Public but distributed via Kaggle:
#   kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022
# Place the CSV(s) under gater-icaif/.data/ (e.g. .data/baf/Base.csv).
_BAF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        ".data", "baf")
_BAF_LABEL = "fraud_bool"
_BAF_TIME = "month"


def load_baf(variant: str = "Base") -> Tuple[pd.DataFrame, DatasetSpec]:
    """Load a BAF variant. Roles inferred from dtype (robust to schema tweaks):
    categorical = object columns; numeric = the rest; label + month excluded from
    features. Constant columns (e.g. device_fraud_count in Base) are dropped."""
    path = os.path.join(_BAF_DIR, f"{variant}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BAF variant not found at {path}. Download once with:\n"
            f"  kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022 "
            f"-p {_BAF_DIR} --unzip")
    df = pd.read_csv(path)
    if _BAF_LABEL not in df.columns:
        raise ValueError(f"{path}: missing label column {_BAF_LABEL!r}")

    exclude = {_BAF_LABEL, _BAF_TIME}
    # drop constant feature columns (uninformative + break QuantileTransformer)
    const = [c for c in df.columns if c not in exclude and df[c].nunique() <= 1]
    feats = [c for c in df.columns if c not in exclude and c not in const]
    categorical = [c for c in feats if df[c].dtype == object]
    numeric = [c for c in feats if c not in categorical]
    for c in categorical:
        df[c] = df[c].astype(str)
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    card = {c: int(df[c].nunique()) for c in categorical}
    spec = DatasetSpec(f"baf_{variant.lower()}", numeric, categorical, _BAF_LABEL,
                       n_rows=len(df), fraud_rate=float(df[_BAF_LABEL].mean()),
                       card=card, time_col=_BAF_TIME if _BAF_TIME in df.columns else None)
    return df, spec


# --- TabFormer: IBM synthetic credit-card transactions (Padhi et al. 2021) ----
# The canonical TabBERT/TabGPT dataset. ~24M rows, GENUINE high-cardinality cats
# (merchant name/city/zip/MCC) -> the dataset where C2's embedding-excision arm bites,
# and the direct TabBERT comparison. Ultra-rare fraud (~0.12%).
#   IBM/TabFormer: card_transaction.v1.csv  (place under .data/tabformer/)
#
# Because it is huge, subsampling is done INSIDE the loader (fixed seed) so every
# caller — baselines AND gater_runner — gets the SAME rows and embeddings align:
#   TABFORMER_SAMPLE=1000000   # random sample of N rows (seed 42), OR
#   TABFORMER_NROWS=1000000    # first N rows (deterministic head)
_TABFORMER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              ".data", "tabformer")
_TABFORMER_SEED = 42


def _collapse_rare(s: pd.Series, min_freq: int = 100) -> pd.Series:
    """Collapse categories appearing < min_freq into 'OTHER' ."""
    vc = s.value_counts()
    rare = set(vc[vc < min_freq].index)
    return s.where(~s.isin(rare), "OTHER")


def _tabformer_file() -> str:
    for ext in ("card_transaction.v1.csv", "card_transaction.v1.parquet",
                "transactions.csv", "transactions.parquet"):
        p = os.path.join(_TABFORMER_DIR, ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"TabFormer file not found in {_TABFORMER_DIR}. Get card_transaction.v1.csv from "
        f"the IBM/TabFormer repo (https://github.com/IBM/TabFormer) and place it there.")


def load_tabformer(min_freq: int = 100) -> Tuple[pd.DataFrame, DatasetSpec]:
    # NOTE: the <min_freq -> OTHER collapse is SAMPLE-SIZE DEPENDENT. At full 24M
    # scale, thousands of merchants clear 100 occurrences and stay high-cardinality
    # (-> embeddings, which is the point). On a small local subsample almost every
    # merchant collapses to OTHER, erasing the cardinality that makes GATER/C2
    # interesting -- so run the high-card story at full scale (cluster). For local
    # experiments lower the threshold via TABFORMER_MINFREQ.
    min_freq = int(os.environ.get("TABFORMER_MINFREQ", min_freq))
    path = _tabformer_file()
    nrows = int(os.environ["TABFORMER_NROWS"]) if os.environ.get("TABFORMER_NROWS") else None
    if nrows is not None and nrows <= 0:
        raise ValueError("TABFORMER_NROWS must be positive")
    reader = pd.read_parquet if path.endswith(".parquet") else pd.read_csv
    raw = reader(path, nrows=nrows) if not path.endswith(".parquet") else reader(path)
    if path.endswith(".parquet") and nrows is not None:
        raw = raw.head(nrows).copy()
    sample = int(os.environ["TABFORMER_SAMPLE"]) if os.environ.get("TABFORMER_SAMPLE") else None
    if sample is not None and sample <= 0:
        raise ValueError("TABFORMER_SAMPLE must be positive")
    if sample and sample < len(raw):
        raw = raw.sample(sample, random_state=_TABFORMER_SEED).reset_index(drop=True)

    df = pd.DataFrame()
    df["amount"] = (raw["Amount"].astype(str).str.replace("$", "", regex=False)
                     .str.replace(",", "", regex=False).astype(float))
    df["is_fraud"] = (raw["Is Fraud?"].astype(str).str.strip().str.lower() == "yes").astype(int)
    df["hour"] = raw["Time"].astype(str).str.slice(0, 2)
    dt = pd.to_datetime(dict(year=raw["Year"], month=raw["Month"], day=raw["Day"]),
                        errors="coerce")
    df["weekday"] = dt.dt.dayofweek.astype("Int64").astype(str)
    df["month_of_year"] = raw["Month"].astype(str)
    df["use_chip"] = raw["Use Chip"].astype(str).fillna("missing")
    df["errors"] = raw.get("Errors?", pd.Series(index=raw.index)).astype(str).where(
        raw.get("Errors?", pd.Series(index=raw.index)).notna(), "none")
    df["mcc"] = raw["MCC"].astype(str)
    df["merchant_state"] = raw["Merchant State"].astype(object).where(
        raw["Merchant State"].notna(), "missing").astype(str)
    # high-cardinality -> frequency-collapse rare to OTHER (embedding-bound)
    df["merchant_name"] = _collapse_rare(raw["Merchant Name"].astype(str), min_freq)
    df["merchant_city"] = _collapse_rare(raw["Merchant City"].astype(str), min_freq)
    df["zip"] = _collapse_rare(raw["Zip"].astype(object).where(raw["Zip"].notna(),
                                "missing").astype(str), min_freq)
    df["period"] = (raw["Year"].astype(int) * 12 + raw["Month"].astype(int))  # drift key

    numeric = ["amount"]
    categorical = ["use_chip", "errors", "mcc", "merchant_state", "hour", "weekday",
                   "month_of_year", "merchant_name", "merchant_city", "zip"]
    card = {c: int(df[c].nunique()) for c in categorical}
    spec = DatasetSpec("tabformer", numeric, categorical, "is_fraud",
                       n_rows=len(df), fraud_rate=float(df["is_fraud"].mean()),
                       card=card, time_col="period")
    return df, spec


LOADERS = {"ieee_cis": load_ieee_cis, "ieee_cis_hc": load_ieee_cis_highcard,
           "ulb_creditcard": load_creditcard,
           "baf": load_baf, "tabformer": load_tabformer}


def load(name: str) -> Tuple[pd.DataFrame, DatasetSpec]:
    if name not in LOADERS:
        raise KeyError(f"unknown dataset {name!r}; have {list(LOADERS)}")
    return LOADERS[name]()


if __name__ == "__main__":
    import sys
    for nm in (sys.argv[1:] or ["ieee_cis"]):
        try:
            d, s = load(nm)
            print(f"{s.name}: {d.shape} | features={len(s.features)} "
                  f"| fraud={s.fraud_rate:.4f} | cat_card={s.card}")
        except Exception as e:  # noqa: BLE001
            print(f"{nm}: {type(e).__name__}: {e}")



def describe_population(frame, spec):
    from experiment_io import dataset_fingerprint
    return replace(spec, n_rows=len(frame), fraud_rate=float(frame[spec.label].mean()),
                   card={column: int(frame[column].nunique()) for column in spec.categorical},
                   row_fingerprint=dataset_fingerprint(frame, spec))
