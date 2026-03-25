"""Cleaning-first data pipeline for the Olist AI Analyst project.

This module enforces mandatory pre-EDA cleaning and quality gates:
- cleans all raw Olist tables
- writes typed Parquet outputs into data/processed
- writes data quality and referential integrity reports
- blocks downstream EDA if critical checks fail
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"


@dataclass
class TableMetrics:
    """Container for per-table cleaning metrics used in quality reports."""

    rows_raw: int
    rows_clean: int
    duplicates_removed: int = 0
    null_key_counts: dict[str, int] = field(default_factory=dict)
    invalid_datetime_counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(file_name: str, dtype: dict[str, str] | None = None) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / file_name, dtype=dtype, low_memory=False)


def _strip_normalize(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    result = result.mask(result == "", pd.NA)
    return result


def _normalize_city(series: pd.Series) -> pd.Series:
    return _strip_normalize(series).str.title()


def _normalize_state(series: pd.Series) -> pd.Series:
    return _strip_normalize(series).str.upper()


def _parse_datetime(df: pd.DataFrame, col: str) -> int:
    non_null_before = int(df[col].notna().sum()) if col in df.columns else 0
    df[col] = pd.to_datetime(df[col], errors="coerce")
    non_null_after = int(df[col].notna().sum())
    return max(non_null_before - non_null_after, 0)


def _ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _record_null_keys(df: pd.DataFrame, keys: list[str]) -> dict[str, int]:
    return {k: int(df[k].isna().sum()) for k in keys}


def clean_customers() -> tuple[pd.DataFrame, TableMetrics]:
    df = _read_csv(
        "olist_customers_dataset.csv",
        dtype={"customer_zip_code_prefix": "string"},
    )
    rows_raw = len(df)
    df["customer_city"] = _normalize_city(df["customer_city"])
    df["customer_state"] = _normalize_state(df["customer_state"])
    df["customer_zip_code_prefix"] = _strip_normalize(df["customer_zip_code_prefix"])

    before = len(df)
    df = df.drop_duplicates(subset=["customer_id"])
    duplicates_removed = before - len(df)

    metrics = TableMetrics(
        rows_raw=rows_raw,
        rows_clean=len(df),
        duplicates_removed=duplicates_removed,
        null_key_counts=_record_null_keys(df, ["customer_id"]),
    )
    return df, metrics


def clean_orders() -> tuple[pd.DataFrame, TableMetrics]:
    df = _read_csv("olist_orders_dataset.csv")
    rows_raw = len(df)

    datetime_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]

    invalid_datetime_counts: dict[str, int] = {}
    for col in datetime_cols:
        invalid_datetime_counts[col] = _parse_datetime(df, col)

    df["order_status"] = _strip_normalize(df["order_status"]).str.lower()

    before = len(df)
    df = df.drop_duplicates(subset=["order_id"])
    duplicates_removed = before - len(df)

    # Track sequence issues but do not drop rows.
    sequence_ok = (
        (df["order_approved_at"].isna() | (df["order_purchase_timestamp"] <= df["order_approved_at"]))
        & (
            df["order_delivered_carrier_date"].isna()
            | (df["order_approved_at"].isna() | (df["order_approved_at"] <= df["order_delivered_carrier_date"]))
        )
        & (
            df["order_delivered_customer_date"].isna()
            | (
                df["order_delivered_carrier_date"].isna()
                | (df["order_delivered_carrier_date"] <= df["order_delivered_customer_date"])
            )
        )
    )
    df["temporal_violation"] = ~sequence_ok

    metrics = TableMetrics(
        rows_raw=rows_raw,
        rows_clean=len(df),
        duplicates_removed=duplicates_removed,
        null_key_counts=_record_null_keys(df, ["order_id", "customer_id"]),
        invalid_datetime_counts=invalid_datetime_counts,
    )
    metrics.notes.append(f"temporal_violation_rows={int(df['temporal_violation'].sum())}")
    return df, metrics


def clean_order_items() -> tuple[pd.DataFrame, TableMetrics]:
    df = _read_csv("olist_order_items_dataset.csv")
    rows_raw = len(df)

    invalid_datetime_counts = {"shipping_limit_date": _parse_datetime(df, "shipping_limit_date")}

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["freight_value"] = pd.to_numeric(df["freight_value"], errors="coerce")
    df["is_negative_price"] = df["price"] < 0
    df["is_negative_freight"] = df["freight_value"] < 0

    price_p99 = df["price"].quantile(0.99)
    freight_p99 = df["freight_value"].quantile(0.99)
    df["is_price_outlier"] = df["price"] > price_p99
    df["is_freight_outlier"] = df["freight_value"] > freight_p99

    before = len(df)
    df = df.drop_duplicates(subset=["order_id", "order_item_id"])
    duplicates_removed = before - len(df)

    metrics = TableMetrics(
        rows_raw=rows_raw,
        rows_clean=len(df),
        duplicates_removed=duplicates_removed,
        null_key_counts=_record_null_keys(df, ["order_id", "product_id", "seller_id"]),
        invalid_datetime_counts=invalid_datetime_counts,
    )
    metrics.notes.append(f"negative_price_rows={int(df['is_negative_price'].sum())}")
    metrics.notes.append(f"negative_freight_rows={int(df['is_negative_freight'].sum())}")
    return df, metrics


def clean_order_payments() -> tuple[pd.DataFrame, TableMetrics]:
    df = _read_csv("olist_order_payments_dataset.csv")
    rows_raw = len(df)

    df["payment_type"] = _strip_normalize(df["payment_type"]).str.lower()
    df["payment_installments"] = pd.to_numeric(df["payment_installments"], errors="coerce")
    df["payment_value"] = pd.to_numeric(df["payment_value"], errors="coerce")

    df["invalid_installments"] = df["payment_installments"] < 1
    df["invalid_payment_value"] = df["payment_value"] <= 0

    before = len(df)
    df = df.drop_duplicates(subset=["order_id", "payment_sequential", "payment_type", "payment_value"])
    duplicates_removed = before - len(df)

    metrics = TableMetrics(
        rows_raw=rows_raw,
        rows_clean=len(df),
        duplicates_removed=duplicates_removed,
        null_key_counts=_record_null_keys(df, ["order_id"]),
    )
    metrics.notes.append(f"invalid_installments_rows={int(df['invalid_installments'].sum())}")
    metrics.notes.append(f"invalid_payment_value_rows={int(df['invalid_payment_value'].sum())}")
    return df, metrics


def clean_order_reviews() -> tuple[pd.DataFrame, TableMetrics]:
    df = _read_csv("olist_order_reviews_dataset.csv")
    rows_raw = len(df)

    invalid_datetime_counts = {
        "review_creation_date": _parse_datetime(df, "review_creation_date"),
        "review_answer_timestamp": _parse_datetime(df, "review_answer_timestamp"),
    }

    df["review_comment_title"] = _strip_normalize(df["review_comment_title"])
    df["review_comment_message"] = _strip_normalize(df["review_comment_message"])
    df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")
    df["invalid_review_score"] = ~df["review_score"].isin([1, 2, 3, 4, 5])

    before = len(df)
    df = df.drop_duplicates(subset=["review_id"])
    duplicates_removed = before - len(df)

    metrics = TableMetrics(
        rows_raw=rows_raw,
        rows_clean=len(df),
        duplicates_removed=duplicates_removed,
        null_key_counts=_record_null_keys(df, ["review_id", "order_id"]),
        invalid_datetime_counts=invalid_datetime_counts,
    )
    metrics.notes.append(f"invalid_review_score_rows={int(df['invalid_review_score'].sum())}")
    return df, metrics


def clean_products() -> tuple[pd.DataFrame, TableMetrics]:
    df = _read_csv("olist_products_dataset.csv")
    rows_raw = len(df)

    df["product_category_name"] = _strip_normalize(df["product_category_name"]).str.lower()

    numeric_cols = [
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["invalid_weight"] = df["product_weight_g"].notna() & (df["product_weight_g"] <= 0)
    df["invalid_dimension"] = (
        (df["product_length_cm"].notna() & (df["product_length_cm"] <= 0))
        | (df["product_height_cm"].notna() & (df["product_height_cm"] <= 0))
        | (df["product_width_cm"].notna() & (df["product_width_cm"] <= 0))
    )

    before = len(df)
    df = df.drop_duplicates(subset=["product_id"])
    duplicates_removed = before - len(df)

    metrics = TableMetrics(
        rows_raw=rows_raw,
        rows_clean=len(df),
        duplicates_removed=duplicates_removed,
        null_key_counts=_record_null_keys(df, ["product_id"]),
    )
    metrics.notes.append(f"invalid_weight_rows={int(df['invalid_weight'].sum())}")
    metrics.notes.append(f"invalid_dimension_rows={int(df['invalid_dimension'].sum())}")
    return df, metrics


def clean_sellers() -> tuple[pd.DataFrame, TableMetrics]:
    df = _read_csv(
        "olist_sellers_dataset.csv",
        dtype={"seller_zip_code_prefix": "string"},
    )
    rows_raw = len(df)

    df["seller_city"] = _normalize_city(df["seller_city"])
    df["seller_state"] = _normalize_state(df["seller_state"])
    df["seller_zip_code_prefix"] = _strip_normalize(df["seller_zip_code_prefix"])

    before = len(df)
    df = df.drop_duplicates(subset=["seller_id"])
    duplicates_removed = before - len(df)

    metrics = TableMetrics(
        rows_raw=rows_raw,
        rows_clean=len(df),
        duplicates_removed=duplicates_removed,
        null_key_counts=_record_null_keys(df, ["seller_id"]),
    )
    return df, metrics


def clean_geolocation() -> tuple[pd.DataFrame, TableMetrics]:
    df = _read_csv(
        "olist_geolocation_dataset.csv",
        dtype={"geolocation_zip_code_prefix": "string"},
    )
    rows_raw = len(df)

    df["geolocation_city"] = _normalize_city(df["geolocation_city"])
    df["geolocation_state"] = _normalize_state(df["geolocation_state"])
    df["geolocation_zip_code_prefix"] = _strip_normalize(df["geolocation_zip_code_prefix"])

    df["geolocation_lat"] = pd.to_numeric(df["geolocation_lat"], errors="coerce")
    df["geolocation_lng"] = pd.to_numeric(df["geolocation_lng"], errors="coerce")
    df["is_invalid_geo"] = (
        (df["geolocation_lat"] < -34)
        | (df["geolocation_lat"] > 6)
        | (df["geolocation_lng"] < -75)
        | (df["geolocation_lng"] > -28)
    )

    metrics = TableMetrics(
        rows_raw=rows_raw,
        rows_clean=len(df),
        null_key_counts=_record_null_keys(df, ["geolocation_zip_code_prefix"]),
    )
    metrics.notes.append(f"invalid_geo_rows={int(df['is_invalid_geo'].sum())}")
    return df, metrics


def _orphan_stats(child: pd.DataFrame, child_key: str, parent: pd.DataFrame, parent_key: str) -> dict[str, Any]:
    valid_child = child[child[child_key].notna()]
    orphan_mask = ~valid_child[child_key].isin(parent[parent_key])
    orphan_count = int(orphan_mask.sum())
    checked = int(len(valid_child))
    orphan_rate = float(orphan_count / checked) if checked else 0.0
    return {
        "child_key": child_key,
        "parent_key": parent_key,
        "checked_rows": checked,
        "orphan_count": orphan_count,
        "orphan_rate": round(orphan_rate, 6),
    }


def _to_metrics_payload(metrics: dict[str, TableMetrics]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name, m in metrics.items():
        payload[name] = {
            "rows_raw": m.rows_raw,
            "rows_clean": m.rows_clean,
            "duplicates_removed": m.duplicates_removed,
            "null_key_counts": m.null_key_counts,
            "invalid_datetime_counts": m.invalid_datetime_counts,
            "notes": m.notes,
        }
    return payload


def run_pipeline() -> dict[str, Any]:
    """Execute full pre-EDA cleaning workflow and return report payload."""

    _ensure_dirs()

    tables: dict[str, pd.DataFrame] = {}
    metrics: dict[str, TableMetrics] = {}

    tables["customers"], metrics["customers"] = clean_customers()
    tables["orders"], metrics["orders"] = clean_orders()
    tables["order_items"], metrics["order_items"] = clean_order_items()
    tables["order_payments"], metrics["order_payments"] = clean_order_payments()
    tables["order_reviews"], metrics["order_reviews"] = clean_order_reviews()
    tables["products"], metrics["products"] = clean_products()
    tables["sellers"], metrics["sellers"] = clean_sellers()
    tables["geolocation"], metrics["geolocation"] = clean_geolocation()

    referential_report = {
        "generated_at_utc": _utc_now_iso(),
        "checks": {
            "orders.customer_id -> customers.customer_id": _orphan_stats(
                tables["orders"], "customer_id", tables["customers"], "customer_id"
            ),
            "order_items.order_id -> orders.order_id": _orphan_stats(
                tables["order_items"], "order_id", tables["orders"], "order_id"
            ),
            "order_items.product_id -> products.product_id": _orphan_stats(
                tables["order_items"], "product_id", tables["products"], "product_id"
            ),
            "order_items.seller_id -> sellers.seller_id": _orphan_stats(
                tables["order_items"], "seller_id", tables["sellers"], "seller_id"
            ),
            "order_payments.order_id -> orders.order_id": _orphan_stats(
                tables["order_payments"], "order_id", tables["orders"], "order_id"
            ),
            "order_reviews.order_id -> orders.order_id": _orphan_stats(
                tables["order_reviews"], "order_id", tables["orders"], "order_id"
            ),
        },
    }

    quality_report: dict[str, Any] = {
        "generated_at_utc": _utc_now_iso(),
        "tables": _to_metrics_payload(metrics),
        "critical_failures": [],
        "warnings": [],
    }

    # Critical gates: null keys, datetime parse failures.
    for table_name, table_metrics in metrics.items():
        for key, null_count in table_metrics.null_key_counts.items():
            if null_count > 0:
                quality_report["critical_failures"].append(
                    f"{table_name}.{key} has {null_count} null values"
                )
        for col, invalid_count in table_metrics.invalid_datetime_counts.items():
            if invalid_count > 0:
                quality_report["critical_failures"].append(
                    f"{table_name}.{col} has {invalid_count} unparseable datetime values"
                )

    # Referential threshold: fail when orphan rate exceeds 0.1% for required relationships.
    for rel_name, rel_stats in referential_report["checks"].items():
        orphan_rate = float(rel_stats["orphan_rate"])
        if orphan_rate > 0.001:
            quality_report["critical_failures"].append(
                f"{rel_name} orphan_rate {orphan_rate:.4%} exceeds 0.10%"
            )
        elif rel_stats["orphan_count"] > 0:
            quality_report["warnings"].append(
                f"{rel_name} has {rel_stats['orphan_count']} orphans ({orphan_rate:.4%})"
            )

    # Write cleaned outputs regardless of failure to support diagnosis.
    output_names = {
        "customers": "olist_customers_clean.parquet",
        "orders": "olist_orders_clean.parquet",
        "order_items": "olist_order_items_clean.parquet",
        "order_payments": "olist_order_payments_clean.parquet",
        "order_reviews": "olist_order_reviews_clean.parquet",
        "products": "olist_products_clean.parquet",
        "sellers": "olist_sellers_clean.parquet",
        "geolocation": "olist_geolocation_clean.parquet",
    }
    for table_name, output_file in output_names.items():
        tables[table_name].to_parquet(PROCESSED_DIR / output_file, index=False)

    with (PROCESSED_DIR / "data_quality_report.json").open("w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    with (PROCESSED_DIR / "referential_integrity_report.json").open("w", encoding="utf-8") as f:
        json.dump(referential_report, f, indent=2)

    # Duplicate reports to reports/ for BI/reporting convenience.
    with (REPORTS_DIR / "data_quality_report.json").open("w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    with (REPORTS_DIR / "referential_integrity_report.json").open("w", encoding="utf-8") as f:
        json.dump(referential_report, f, indent=2)

    return {
        "quality_report": quality_report,
        "referential_report": referential_report,
    }


def main() -> None:
    """Run pipeline and stop on critical quality failures."""

    result = run_pipeline()
    failures = result["quality_report"]["critical_failures"]
    if failures:
        failure_text = "\n - ".join([""] + failures)
        raise RuntimeError(f"Pre-EDA quality gate failed:{failure_text}")

    print("Pre-EDA quality gate passed. Cleaned datasets are ready in data/processed.")


if __name__ == "__main__":
    main()
