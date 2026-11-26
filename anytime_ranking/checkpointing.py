"""Atomic query checkpoints with explicit JSON serialization."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
from time import sleep

import numpy as np

from .ranking import PosteriorSummary, ReportCertificate


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"cannot serialize checkpoint value: {type(value)}")


def atomic_json(path, value, *, durable=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    encoded = json.dumps(value, ensure_ascii=False, default=_json_value, allow_nan=True)
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        if durable:
            os.fsync(handle.fileno())
    for attempt in range(7):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            # Windows readers or virus scanners can briefly hold the target.
            # Keep the previous checkpoint intact until replacement succeeds.
            if os.name != "nt" or attempt == 6:
                raise
            sleep(0.02 * 2**attempt)


def restore_summary(value):
    if value is None:
        return None
    values = dict(value)
    values["labels"] = np.asarray(values["labels"], dtype=np.int64)
    certificate = values.get("report_certificate")
    if certificate is not None:
        certificate = dict(certificate)
        for name in ("labels", "parameters", "gap_by_model"):
            certificate[name] = np.asarray(certificate[name], dtype=np.int64)
        for name in ("cut_points", "evaluator_id"):
            certificate[name] = tuple(certificate[name])
        values["report_certificate"] = ReportCertificate(**certificate)
    return PosteriorSummary(**values)
