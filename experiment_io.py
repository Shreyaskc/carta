"""Scientific artifact identity and compatibility checks for new experiment outputs."""
import hashlib
import json
from pathlib import Path
import os
import tempfile
import warnings

import numpy as np
import pandas as pd


def dataset_fingerprint(frame, spec):
    columns = list(spec.features)
    if spec.time_col and spec.time_col not in columns:
        columns.append(spec.time_col)
    values = frame[columns]
    digest = hashlib.sha256()
    digest.update(json.dumps([(name, str(values[name].dtype)) for name in columns]).encode())
    digest.update(pd.util.hash_pandas_object(values, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def evaluation_fingerprint(frame, spec):
    """Bind supervised scores to both feature rows and their ordered targets."""
    digest = hashlib.sha256(dataset_fingerprint(frame, spec).encode())
    digest.update(str(frame[spec.label].dtype).encode())
    digest.update(pd.util.hash_pandas_object(frame[spec.label], index=False).to_numpy().tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_embedding(path, embedding, spec):
    embedding = np.asarray(embedding)
    if (embedding.ndim != 2 or not all(embedding.shape)
            or not np.issubdtype(embedding.dtype, np.number) or not np.isfinite(embedding).all()):
        raise ValueError("Embedding must be a nonempty finite numeric 2D array")
    if not spec.row_fingerprint or spec.n_rows != len(embedding):
        raise ValueError("Embedding requires matching population metadata before saving")
    path = Path(path)
    if path.suffix != ".npy":
        path = Path(str(path) + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"dataset": spec.name, "n_rows": len(embedding), "shape": list(embedding.shape),
                "row_fingerprint": spec.row_fingerprint, "protocol": spec.encoding_protocol}
    # Serialize before replacing either file. Two-file replacement is not atomic,
    # but any interrupted mismatch fails checksum verification on the next read.
    temporary = []
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            array_path = Path(handle.name)
            temporary.append(array_path)
            np.save(handle, embedding, allow_pickle=False)
        metadata["embedding_sha256"] = file_digest(array_path)
        payload = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
            metadata_path = Path(handle.name)
            temporary.append(metadata_path)
            handle.write(payload)
        os.replace(array_path, path)
        os.replace(metadata_path, path.with_suffix(".metadata.json"))
    finally:
        for temporary_path in temporary:
            temporary_path.unlink(missing_ok=True)


def _protocol_matches(actual, expected):
    return isinstance(actual, dict) and all(
        _protocol_matches(actual.get(key), value) if isinstance(value, dict)
        else actual.get(key) == value for key, value in expected.items())


def load_embedding(path, frame, spec, allow_unverified=False, expected_protocol=None, return_metadata=False):
    path = Path(path)
    embedding = np.load(path, allow_pickle=False)
    if (embedding.ndim != 2 or not all(embedding.shape) or len(embedding) != len(frame)
            or not np.issubdtype(embedding.dtype, np.number) or not np.isfinite(embedding).all()):
        raise ValueError("Embedding must be a finite 2D array aligned in row count")
    sidecar = path.with_suffix(".metadata.json")
    if not sidecar.exists():
        if not allow_unverified:
            raise ValueError("Embedding row identity is unverified: metadata sidecar missing")
        warnings.warn("Embedding row identity is unverified; row count alone is insufficient", RuntimeWarning)
        return (embedding, {"verification": "unverified legacy artifact"}) if return_metadata else embedding
    metadata = json.loads(sidecar.read_text())
    if metadata.get("dataset") != spec.name or metadata.get("row_fingerprint") != dataset_fingerprint(frame, spec):
        raise ValueError("Embedding dataset or ordered row fingerprint mismatch")
    if metadata.get("shape") != list(embedding.shape) or metadata.get("n_rows") != len(embedding):
        raise ValueError("Embedding shape differs from recorded metadata")
    checksum = metadata.get("embedding_sha256")
    if checksum is None:
        if not allow_unverified:
            raise ValueError("Embedding content is unverified: checksum missing")
        warnings.warn("Embedding content is unverified: checksum missing", RuntimeWarning)
        metadata["verification"] = "unverified legacy content"
    elif checksum != file_digest(path):
        raise ValueError("Embedding content checksum mismatch")
    else:
        metadata["verification"] = "verified rows and content"
    if expected_protocol and not _protocol_matches(metadata.get("protocol", {}), expected_protocol):
        raise ValueError("Embedding protocol differs from the requested representation")
    return (embedding, metadata) if return_metadata else embedding


def require_matching_protocol(existing, dataset, seeds, epochs, population_fingerprint):
    if existing.get("methods") and any(existing.get(key) != value for key, value in
                                       (("dataset", dataset), ("seeds", list(seeds)), ("epochs", epochs),
                                        ("evaluation_fingerprint", population_fingerprint))):
        raise ValueError("Existing comparison methods use a different or unknown protocol; select a separate output directory")


def validate_reference_overrides(overrides):
    conflicting = set(overrides) & {"random_state", "epochs", "batch_size"}
    if conflicting:
        raise ValueError("Use SEEDS/EPOCHS/BATCH instead of conflicting TRAIN_OVERRIDES fields")


def require_baseline_compatibility(existing, frame, spec, seeds):
    if existing and (existing.get("dataset") != spec.name or existing.get("seeds") != list(seeds)
                     or existing.get("row_fingerprint") != dataset_fingerprint(frame, spec)
                     or existing.get("evaluation_fingerprint") != evaluation_fingerprint(frame, spec)):
        raise ValueError("Existing baseline population/probe seeds differ or are unverified; use a separate output directory")
