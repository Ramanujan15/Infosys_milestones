import pandas as pd

def load_and_preprocess(file):
    """Load CSV, ensure timestamp column, and resample data hourly."""
    df = pd.read_csv(file)

    if "timestamp" not in df.columns:
        raise ValueError("❌ 'timestamp' column not found in dataset.")

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp")

    # Separate numeric and non-numeric
    numeric_df = df.select_dtypes(include="number").resample("H").mean()
    non_numeric_df = df.select_dtypes(exclude="number").resample("H").ffill()

    # Merge back
    df_resampled = pd.concat([numeric_df, non_numeric_df], axis=1).reset_index()

    return df_resampled
