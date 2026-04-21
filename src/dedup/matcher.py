"""
Patient deduplication using fuzzy matching + record linkage.
Identifies the same patient appearing under different spellings/IDs across systems.
"""
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    patient_id_a: str
    patient_id_b: str
    name_score: float
    dob_match: bool
    phone_match: bool
    composite_score: float
    is_duplicate: bool


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return " ".join(name.lower().strip().split())


def normalize_phone(phone: str) -> str:
    if not isinstance(phone, str):
        return ""
    digits = "".join(c for c in phone if c.isdigit())
    if digits.startswith("234") and len(digits) == 13:
        digits = "0" + digits[3:]
    return digits[-10:] if len(digits) >= 10 else digits


def normalize_dob(dob) -> Optional[str]:
    if pd.isna(dob):
        return None
    try:
        return pd.to_datetime(dob).strftime("%Y-%m-%d")
    except Exception:
        return None


def blocking_key(row: pd.Series) -> str:
    name = normalize_name(str(row.get("full_name", "")))
    parts = name.split()
    first_initial = parts[0][0] if parts else "x"
    birth_year = ""
    try:
        birth_year = str(pd.to_datetime(row.get("date_of_birth", "")).year)
    except Exception:
        pass
    return f"{first_initial}_{birth_year}"


class PatientDeduplicator:
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold

    def compute_match_score(self, row_a: pd.Series, row_b: pd.Series) -> MatchResult:
        name_a = normalize_name(str(row_a.get("full_name", "")))
        name_b = normalize_name(str(row_b.get("full_name", "")))
        name_score = fuzz.token_sort_ratio(name_a, name_b) / 100.0

        dob_a = normalize_dob(row_a.get("date_of_birth"))
        dob_b = normalize_dob(row_b.get("date_of_birth"))
        dob_match = (dob_a is not None and dob_a == dob_b)

        phone_a = normalize_phone(str(row_a.get("phone", "")))
        phone_b = normalize_phone(str(row_b.get("phone", "")))
        phone_match = bool(phone_a and phone_a == phone_b)

        composite = (
            name_score * 0.5 +
            float(dob_match) * 0.35 +
            float(phone_match) * 0.15
        )

        return MatchResult(
            patient_id_a=str(row_a.get("patient_id", "")),
            patient_id_b=str(row_b.get("patient_id", "")),
            name_score=name_score,
            dob_match=dob_match,
            phone_match=phone_match,
            composite_score=composite,
            is_duplicate=composite >= self.threshold,
        )

    def find_duplicates(self, df: pd.DataFrame) -> list[MatchResult]:
        df = df.copy()
        df["_blocking_key"] = df.apply(blocking_key, axis=1)
        duplicates = []

        for block_key, group in df.groupby("_blocking_key"):
            if len(group) < 2:
                continue
            indices = group.index.tolist()
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    row_a = group.loc[indices[i]]
                    row_b = group.loc[indices[j]]
                    result = self.compute_match_score(row_a, row_b)
                    if result.is_duplicate:
                        duplicates.append(result)

        logger.info(f"Found {len(duplicates)} duplicate pairs across {df['_blocking_key'].nunique()} blocks")
        return duplicates

    def merge_records(self, df: pd.DataFrame, duplicates: list[MatchResult]) -> pd.DataFrame:
        id_map = {}
        for match in duplicates:
            canonical = min(match.patient_id_a, match.patient_id_b)
            id_map[match.patient_id_a] = canonical
            id_map[match.patient_id_b] = canonical

        df = df.copy()
        df["canonical_patient_id"] = df["patient_id"].apply(lambda pid: id_map.get(str(pid), str(pid)))
        deduped = df.sort_values("created_at", ascending=False).drop_duplicates(subset=["canonical_patient_id"])
        logger.info(f"Records reduced from {len(df)} to {len(deduped)} after deduplication")
        return deduped

    def generate_master_id(self, row: pd.Series) -> str:
        key = f"{normalize_name(str(row.get('full_name', '')))}{normalize_dob(row.get('date_of_birth', ''))}"
        return "MRN-" + hashlib.sha256(key.encode()).hexdigest()[:12].upper()
