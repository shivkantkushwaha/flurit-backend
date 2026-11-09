"""Dataset Profiler for Flurit Onboarding.

Provides secure ingestion, structural verification, statistical profiling,
and type inference for uploaded CSV customer datasets without leaking PII.
"""

import io
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from app.core.config import get_settings
from app.core.logging import get_logger
from app.onboarding.schemas import (
    ColumnProfile,
    DatasetProfile,
    DateRange,
    InferredDataType,
)

logger = get_logger("flurit.onboarding.profiler")

# Regex to sanitize filenames against directory traversal and dangerous characters
SAFE_FILENAME_REGEX = re.compile(r"[^a-zA-Z0-9_\-\.]")
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")

# Keywords identifying candidate identifier and target columns
ID_KEYWORDS = ("id", "key", "uuid", "code", "account", "sub", "ticket", "user")
TARGET_KEYWORDS = (
    "churn",
    "cancelled",
    "canceled",
    "churned",
    "attrition",
    "target",
    "label",
    "is_churn",
)
DATE_KEYWORDS = (
    "date",
    "time",
    "timestamp",
    "created_at",
    "closed_at",
    "start_date",
    "end_date",
    "signup_date",
    "churn_date",
    "event_time",
)


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename to prevent directory traversal and injection."""
    base_name = Path(filename).name
    # Strip null bytes
    base_name = base_name.replace("\x00", "")
    sanitized = SAFE_FILENAME_REGEX.sub("_", base_name)
    if not sanitized or sanitized.startswith("."):
        return f"dataset_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    if not sanitized.lower().endswith(".csv"):
        sanitized = f"{sanitized}.csv"
    return sanitized


class CSVProfiler:
    """Secure memory-safe CSV profiling engine."""

    def __init__(
        self,
        max_file_size_bytes: int | None = None,
        max_row_count: int | None = None,
        max_column_count: int | None = None,
    ) -> None:
        settings = get_settings()
        self.max_file_size_bytes = (
            max_file_size_bytes
            if max_file_size_bytes is not None
            else settings.ONBOARDING_MAX_FILE_SIZE_BYTES
        )
        self.max_row_count = (
            max_row_count if max_row_count is not None else settings.ONBOARDING_MAX_ROW_COUNT
        )
        self.max_column_count = (
            max_column_count if max_column_count is not None else settings.ONBOARDING_MAX_COLUMNS
        )

    def profile_bytes(self, raw_bytes: bytes, original_filename: str) -> DatasetProfile:
        """Profile raw CSV byte payload with size, encoding, and schema validation."""
        safe_name = sanitize_filename(original_filename)
        size_bytes = len(raw_bytes)

        if size_bytes == 0:
            raise ValueError(f"Uploaded file '{safe_name}' is empty (0 bytes).")

        if size_bytes > self.max_file_size_bytes:
            max_mb = self.max_file_size_bytes / (1024 * 1024)
            actual_mb = round(size_bytes / (1024 * 1024), 2)
            raise ValueError(
                f"File '{safe_name}' ({actual_mb} MB) exceeds maximum allowed upload size ({max_mb:.0f} MB)."
            )

        # Decode content with encoding fallback
        text_content = self._decode_bytes(raw_bytes, safe_name)

        # Parse CSV into pandas dataframe
        try:
            df = pd.read_csv(
                io.StringIO(text_content),
                low_memory=False,
                encoding_errors="replace",
            )
        except pd.errors.EmptyDataError as exc:
            raise ValueError(
                f"Uploaded file '{safe_name}' contains no data or parseable columns."
            ) from exc
        except Exception as exc:
            logger.warning("CSV parsing error", extra={"file_name": safe_name, "error": str(exc)})
            raise ValueError(
                f"Malformed CSV in '{safe_name}': Unable to parse tabular records."
            ) from exc

        return self.profile_dataframe(df, file_name=safe_name, file_size_bytes=size_bytes)

    def profile_dataframe(
        self, df: pd.DataFrame, file_name: str, file_size_bytes: int = 0
    ) -> DatasetProfile:
        """Generate statistical and structural profile from a parsed DataFrame."""
        row_count = len(df)
        column_count = len(df.columns)

        if row_count > self.max_row_count:
            raise ValueError(
                f"File '{file_name}' contains {row_count:,} rows, exceeding the limit of {self.max_row_count:,} rows."
            )

        if column_count > self.max_column_count:
            raise ValueError(
                f"File '{file_name}' contains {column_count} columns, exceeding the limit of {self.max_column_count} columns."
            )

        if column_count == 0:
            raise ValueError(f"File '{file_name}' has 0 columns.")

        # Clean column names (strip whitespace and handle duplicates)
        clean_columns: list[str] = []
        seen_cols: dict[str, int] = {}
        for col in df.columns:
            clean_name = str(col).strip()
            if not clean_name:
                clean_name = "unnamed_column"
            if clean_name in seen_cols:
                seen_cols[clean_name] += 1
                clean_name = f"{clean_name}_{seen_cols[clean_name]}"
            else:
                seen_cols[clean_name] = 0
            clean_columns.append(clean_name)
        df.columns = pd.Index(clean_columns)

        # Compute duplicate rows
        duplicate_count = int(df.duplicated().sum())

        column_profiles: list[ColumnProfile] = []
        candidate_ids: list[str] = []
        candidate_targets: list[str] = []
        global_min_date: datetime | None = None
        global_max_date: datetime | None = None

        for col_name in df.columns:
            col_series = df[col_name]
            profile, is_id, is_target, col_min_dt, col_max_dt = self._profile_column(
                col_name=col_name,
                series=col_series,
                total_rows=row_count,
            )
            column_profiles.append(profile)

            if is_id:
                candidate_ids.append(col_name)
            if is_target:
                candidate_targets.append(col_name)

            if col_min_dt is not None:
                if global_min_date is None or col_min_dt < global_min_date:
                    global_min_date = col_min_dt
            if col_max_dt is not None:
                if global_max_date is None or col_max_dt > global_max_date:
                    global_max_date = col_max_dt

        # Calculate global date span if datetime columns exist
        global_date_range: DateRange | None = None
        if global_min_date is not None and global_max_date is not None:
            span = (global_max_date.date() - global_min_date.date()).days
            global_date_range = DateRange(
                min_date=global_min_date.date().isoformat(),
                max_date=global_max_date.date().isoformat(),
                span_days=max(0, span),
            )

        logger.info(
            "Completed dataset profile",
            extra={
                "file_name": file_name,
                "rows": row_count,
                "columns": column_count,
                "candidate_ids": candidate_ids,
                "candidate_targets": candidate_targets,
            },
        )

        return DatasetProfile(
            file_name=file_name,
            file_size_bytes=file_size_bytes,
            row_count=row_count,
            column_count=column_count,
            duplicate_row_count=duplicate_count,
            candidate_id_columns=candidate_ids,
            candidate_target_columns=candidate_targets,
            columns=column_profiles,
            date_range=global_date_range,
        )

    def _decode_bytes(self, raw_bytes: bytes, file_name: str) -> str:
        """Decode raw bytes with safe multi-encoding fallback."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for enc in encodings:
            try:
                return raw_bytes.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue

        raise ValueError(
            f"Encoding error for '{file_name}': File encoding could not be determined. "
            "Please upload a standard UTF-8 encoded CSV file."
        )

    def _profile_column(
        self, col_name: str, series: pd.Series, total_rows: int
    ) -> tuple[ColumnProfile, bool, bool, datetime | None, datetime | None]:
        """Profile an individual column, inferring its type, statistics, and role."""
        null_count = int(series.isna().sum())
        null_pct = round((null_count / total_rows * 100.0), 2) if total_rows > 0 else 0.0

        non_null = series.dropna()
        non_null_count = len(non_null)
        unique_count = int(non_null.nunique())
        unique_ratio = round((unique_count / non_null_count), 4) if non_null_count > 0 else 0.0

        col_lower = col_name.lower()

        # Step 1: Infer column type
        inferred_type, min_val_str, max_val_str, date_range, min_dt, max_dt, sample_cats = (
            self._infer_type_and_stats(col_name=col_name, series=non_null)
        )

        # Step 2: Identify candidate primary / foreign keys
        is_candidate_id = False
        if non_null_count > 0 and null_count == 0:
            if unique_ratio >= 0.999:
                if any(kw in col_lower for kw in ID_KEYWORDS) or inferred_type in (
                    InferredDataType.INTEGER,
                    InferredDataType.TEXT,
                ):
                    is_candidate_id = True

        # Step 3: Identify candidate churn targets or status indicators
        is_candidate_target = False
        if any(kw in col_lower for kw in TARGET_KEYWORDS):
            is_candidate_target = True
        elif inferred_type == InferredDataType.BOOLEAN and non_null_count > 0:
            # Check if boolean with plausible churn prevalence (1% to 60%)
            try:
                truthy_count = (
                    non_null.astype(str).str.lower().isin(["1", "true", "yes", "t"]).sum()
                )
                ratio = truthy_count / non_null_count
                if 0.01 <= ratio <= 0.60:
                    is_candidate_target = True
            except Exception:
                pass

        profile = ColumnProfile(
            column_name=col_name,
            inferred_type=inferred_type,
            total_count=total_rows,
            null_count=null_count,
            null_percentage=null_pct,
            unique_count=unique_count,
            unique_ratio=unique_ratio,
            is_candidate_id=is_candidate_id,
            min_value=min_val_str,
            max_value=max_val_str,
            sample_categories=sample_cats,
            date_range=date_range,
        )

        return profile, is_candidate_id, is_candidate_target, min_dt, max_dt

    def _infer_type_and_stats(
        self, col_name: str, series: pd.Series
    ) -> tuple[
        InferredDataType,
        str | None,
        str | None,
        DateRange | None,
        datetime | None,
        datetime | None,
        list[str] | None,
    ]:
        """Infer type, min/max values, date range, and redacted category samples."""
        if series.empty:
            return InferredDataType.UNKNOWN, None, None, None, None, None, None

        col_lower = col_name.lower()
        non_null_count = len(series)

        # Check for Datetime: only non-numeric/non-bool series or explicit datetime dtypes
        is_already_datetime = series.dtype == "datetime64[ns]" or "datetime" in str(series.dtype)
        is_date_named = any(kw in col_lower for kw in DATE_KEYWORDS)
        is_numeric_or_bool = pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(
            series
        )

        if is_already_datetime or (not is_numeric_or_bool and is_date_named):
            try:
                parsed_dates = pd.to_datetime(series, errors="coerce")
                valid_date_count = int(parsed_dates.notna().sum())
                if valid_date_count > 0 and (valid_date_count / non_null_count) >= 0.80:
                    valid_dates = parsed_dates.dropna()
                    min_ts = pd.Timestamp(valid_dates.min())
                    max_ts = pd.Timestamp(valid_dates.max())
                    min_dt = min_ts.to_pydatetime()
                    max_dt = max_ts.to_pydatetime()
                    span = (max_dt.date() - min_dt.date()).days
                    dr = DateRange(
                        min_date=min_dt.date().isoformat(),
                        max_date=max_dt.date().isoformat(),
                        span_days=max(0, span),
                    )
                    return (
                        InferredDataType.DATETIME,
                        min_dt.date().isoformat(),
                        max_dt.date().isoformat(),
                        dr,
                        min_dt,
                        max_dt,
                        None,
                    )
            except Exception:
                pass

        # Check for Boolean
        str_vals = series.astype(str).str.strip().str.lower()
        unique_str_vals = set(str_vals.unique())
        bool_subsets = [
            {"0", "1"},
            {"true", "false"},
            {"yes", "no"},
            {"t", "f"},
            {"0.0", "1.0"},
        ]
        if any(unique_str_vals.issubset(s) for s in bool_subsets) and len(unique_str_vals) <= 2:
            return (
                InferredDataType.BOOLEAN,
                str(min(unique_str_vals)),
                str(max(unique_str_vals)),
                None,
                None,
                None,
                sorted(unique_str_vals),
            )

        # Check for Numeric (Integer or Float)
        if pd.api.types.is_numeric_dtype(series):
            min_val = series.min()
            max_val = series.max()
            min_str = str(min_val) if pd.notna(min_val) else None
            max_str = str(max_val) if pd.notna(max_val) else None

            # Integer check: either integer dtype or all values equal int(val)
            if pd.api.types.is_integer_dtype(series):
                return InferredDataType.INTEGER, min_str, max_str, None, None, None, None

            # Check if float series actually contains only integers
            try:
                if (series.round() == series).all():
                    return InferredDataType.INTEGER, min_str, max_str, None, None, None, None
            except Exception:
                pass

            return InferredDataType.FLOAT, min_str, max_str, None, None, None, None

        # Check for Category vs Text
        unique_count = int(series.nunique())
        unique_ratio = unique_count / non_null_count

        if unique_count <= 25 and (unique_ratio <= 0.25 or non_null_count <= 50):
            # Categorical: extract sanitized sample values
            raw_cats = sorted(series.astype(str).unique()[:10])
            sanitized_cats = [self._sanitize_sample_value(v) for v in raw_cats]
            return (
                InferredDataType.CATEGORY,
                sanitized_cats[0] if sanitized_cats else None,
                sanitized_cats[-1] if sanitized_cats else None,
                None,
                None,
                None,
                sanitized_cats,
            )

        # Fallback to Text
        return InferredDataType.TEXT, None, None, None, None, None, None

    @staticmethod
    def _sanitize_sample_value(value: str) -> str:
        """Redact PII (such as emails or personal IDs) from sample category values."""
        val = str(value).strip()
        if EMAIL_REGEX.search(val):
            return "[REDACTED_EMAIL]"
        if len(val) > 50:
            return f"{val[:47]}..."
        return val
