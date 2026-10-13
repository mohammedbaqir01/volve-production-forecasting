from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer


PRODUCTION_WELL = "15/9-F-12"


def load_production_data(
    workbook_path: str | Path, test_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and chronologically split the target production well.

    Args:
        workbook_path: Path to the Volve Excel workbook. The workbook must contain a
            ``Daily Production Data`` sheet.
        test_fraction: Fraction of the latest sorted records reserved for testing.

    Returns:
        A pair of data frames (training, test), indexed by ``DATEPRD``. Only
        well ``15/9-F-12`` is retained, and the last ``test_fraction`` portion
        of its sorted records forms the test partition.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")

    daily_production_data = pd.read_excel(
        workbook_path, sheet_name="Daily Production Data"
    )
    required_columns = {
        "NPD_WELL_BORE_NAME",
        "FLOW_KIND",
        "DATEPRD",
        "BORE_OIL_VOL",
    }
    missing_columns = required_columns.difference(daily_production_data.columns)
    if missing_columns:
        raise ValueError(
            "Daily Production Data is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    production_only_well_names = (
        daily_production_data.groupby("NPD_WELL_BORE_NAME")["FLOW_KIND"]
        .agg(set)
        .loc[lambda values: values == {"production"}]
    )
    production_records = daily_production_data[
        daily_production_data["NPD_WELL_BORE_NAME"].isin(production_only_well_names.index.tolist())
    ]
    well_df = production_records[
        production_records["NPD_WELL_BORE_NAME"] == PRODUCTION_WELL
    ].copy()
    if well_df.empty:
        raise ValueError(
            f"No production-only records found for well {PRODUCTION_WELL!r}."
        )
    well_df["DATEPRD"] = pd.to_datetime(well_df["DATEPRD"], errors="coerce")
    if well_df["DATEPRD"].isna().any():
        raise ValueError("Production records contain missing or invalid DATEPRD values.")
    if well_df["DATEPRD"].duplicated().any():
        raise ValueError("Production records contain duplicate DATEPRD values.")
    well_df = well_df.sort_values("DATEPRD").set_index("DATEPRD")

    train_test_split_index = int(len(well_df) * (1 - test_fraction))
    if train_test_split_index == 0 or train_test_split_index == len(well_df):
        raise ValueError("The selected well needs enough records for both partitions.")
    train_df = well_df.iloc[:train_test_split_index].copy()
    test_df = well_df.iloc[train_test_split_index:].copy()
    return train_df, test_df


def prepare_production_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fit_end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    """Clean production partitions and impute numeric values without test leakage.

    The function copies its inputs, removes ``BORE_WI_VOL`` when present,
    retains zero-production days, rejects missing targets, and fits a median
    imputer on numeric predictors only. When ``fit_end`` is provided, only
    rows through that date are used to learn medians; later training rows can
    serve as validation data without influencing preprocessing.

    Args:
        train_df: Chronological training partition.
        test_df: Held-out partition with matching numeric feature columns.
        fit_end: Optional last date included when fitting the imputer.

    Returns:
        Cleaned training and test data frames, followed by the fitted
        ``SimpleImputer`` for reuse at inference time.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    for partition_name, frame in (("training", train_df), ("test", test_df)):
        if "BORE_OIL_VOL" not in frame.columns:
            raise ValueError(f"The {partition_name} partition lacks BORE_OIL_VOL.")
        if frame["BORE_OIL_VOL"].isna().any():
            raise ValueError(
                f"The {partition_name} partition contains missing BORE_OIL_VOL values; "
                "target values must not be imputed."
            )
        if "BORE_WI_VOL" in frame.columns:
            frame.drop("BORE_WI_VOL", axis=1, inplace=True)

    numeric_cols = train_df.select_dtypes(include="number").columns.drop(
        "BORE_OIL_VOL", errors="ignore"
    )
    fit_data = train_df if fit_end is None else train_df.loc[:fit_end]
    if fit_data.empty:
        raise ValueError("fit_end selects no rows from the training partition.")
    imputer = SimpleImputer(strategy="median")
    imputer.fit(fit_data[numeric_cols])
    train_df[numeric_cols] = imputer.transform(train_df[numeric_cols])
    test_df[numeric_cols] = imputer.transform(test_df[numeric_cols])
    return train_df, test_df, imputer
