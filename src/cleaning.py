import re
from typing import Tuple

import pandas as pd

BIN_IIN_RE = re.compile(r"^\d{12}$")


def is_valid_iin_bin(value: object) -> bool:
    if pd.isna(value):
        return False
    s = str(value).strip()
    if not BIN_IIN_RE.fullmatch(s):
        return False

    digits = [int(ch) for ch in s]
    weights_1 = list(range(1, 12))
    control = sum(d * w for d, w in zip(digits[:11], weights_1)) % 11

    if control == 10:
        weights_2 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]
        control = sum(d * w for d, w in zip(digits[:11], weights_2)) % 11

    return control != 10 and control == digits[11]


def load_transactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "sender_id": "string",
            "receiver_id": "string",
            "date": "string",
            "description": "string",
            "doc_type": "string",
        },
    )
    df["amount_kzt"] = pd.to_numeric(df["amount_kzt"], errors="coerce")
    return df


def quality_summary(df: pd.DataFrame, stage: str) -> pd.DataFrame:
    temp = df.copy()
    if "sender_id_valid" not in temp.columns:
        temp["sender_id_valid"] = temp["sender_id"].map(is_valid_iin_bin)
    if "receiver_id_valid" not in temp.columns:
        temp["receiver_id_valid"] = temp["receiver_id"].map(is_valid_iin_bin)

    return pd.DataFrame(
        [{
            "stage": stage,
            "rows": len(temp),
            "sender_id_valid_share": round(float(temp["sender_id_valid"].mean()), 4),
            "receiver_id_valid_share": round(float(temp["receiver_id_valid"].mean()), 4),
            "both_ids_valid_share": round(float((temp["sender_id_valid"] & temp["receiver_id_valid"]).mean()), 4),
            "missing_values_share": round(float(temp.isna().mean().mean()), 4),
            "missing_description_share": round(float(temp["description"].isna().mean()), 4),
        }]
    )


def clean_transactions(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    before = quality_summary(df, "before")
    out = df.copy()

    out["sender_id"] = out["sender_id"].astype("string").str.strip()
    out["receiver_id"] = out["receiver_id"].astype("string").str.strip()
    out["description"] = out["description"].astype("string").str.strip()
    out["doc_type"] = out["doc_type"].astype("string").str.strip().str.upper()
    out["amount_kzt"] = pd.to_numeric(out["amount_kzt"], errors="coerce")
    out["date_iso"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True, format="mixed").dt.strftime("%Y-%m-%d")

    out["sender_id_valid"] = out["sender_id"].map(is_valid_iin_bin)
    out["receiver_id_valid"] = out["receiver_id"].map(is_valid_iin_bin)

    dedupe_key = ["sender_id", "receiver_id", "date_iso", "amount_kzt", "description", "doc_type"]
    out = out.drop_duplicates(subset=dedupe_key, keep="first").reset_index(drop=True)

    after = quality_summary(out, "after")
    return out, pd.concat([before, after], ignore_index=True)
