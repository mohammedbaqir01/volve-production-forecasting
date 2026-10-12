from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer


PRODUCTION_WELL = "15/9-F-12"
OIL_COLUMN = "BORE_OIL_VOL"


def load_production_data(workbook_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and chronologically split the target production well.

    Args:
        workbook_path: Path to the Volve Excel workbook. The workbook must contain a
            ``Daily Production Data`` sheet.

    Returns:
        A pair of data frames (training, test), indexed by ``DATEPRD``. Only
        well ``15/9-F-12`` is retained, and the last 15% of its sorted records
        form the test partition.
    """
    workbook_sheets = pd.read_excel(workbook_path, sheet_name=None)
    daily_production_data = workbook_sheets["Daily Production Data"]

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
    well_df["DATEPRD"] = pd.to_datetime(well_df["DATEPRD"])
    well_df = well_df.sort_values("DATEPRD").set_index("DATEPRD")

    train_test_split_index = int(len(well_df) * 0.85)
    train_df = well_df.iloc[:train_test_split_index].copy()
    test_df = well_df.iloc[train_test_split_index:].copy()
    return train_df, test_df


def prepare_production_data(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, SimpleImputer]:
    """Clean production partitions and impute numeric values without test leakage.

    The function copies its inputs, removes ``BORE_WI_VOL`` when present,
    excludes non-positive oil production, drops training rows missing choke
    size, and fits a median imputer on training numeric columns only.

    Args:
        train_df: Chronological training partition.
        test_df: Held-out partition with matching numeric feature columns.

    Returns:
        Cleaned training and test data frames, followed by the fitted
        ``SimpleImputer`` for reuse at inference time.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    for frame in (train_df, test_df):
        if "BORE_WI_VOL" in frame.columns:
            frame.drop("BORE_WI_VOL", axis=1, inplace=True)
        frame.drop(frame[frame[OIL_COLUMN] <= 0].index, inplace=True)

    train_df.dropna(subset=["AVG_CHOKE_SIZE_P"], inplace=True)

    numeric_cols = train_df.select_dtypes(include="number").columns
    imputer = SimpleImputer(strategy="median")
    train_df[numeric_cols] = imputer.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = imputer.transform(test_df[numeric_cols])
    return train_df, test_df, imputer
