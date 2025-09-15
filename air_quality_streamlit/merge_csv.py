import pandas as pd
import glob
import os

files = glob.glob("data/real_*.csv")

dfs = []

for f in files:
    df = pd.read_csv(f)
    print(f"\n📄 File: {f} → Columns: {list(df.columns)}")

    # Extract year from filename (e.g., real_2013.csv → 2013)
    year = os.path.basename(f).split("_")[1].split(".")[0]

    # Create daily date range for that year
    date_range = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")

    # If CSV has fewer/more rows than days, align
    n = min(len(df), len(date_range))
    df = df.iloc[:n].copy()
    df["timestamp"] = date_range[:n]

    # Rename pollution column properly
    df = df.rename(columns={"PM 2.5": "PM2.5"})

    dfs.append(df)

# Combine all years
combined = pd.concat(dfs, ignore_index=True)

# Ensure timestamp is datetime
combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")

# Save
combined.to_csv("data/combined.csv", index=False)

print("\n✅ combined.csv created with timestamp column added")
