import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/combined.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df

df = load_data()

# =========================
# Streamlit Layout
# =========================
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
st.title("🌍 Air Quality Monitoring Dashboard")

st.sidebar.header("Filters")

# Year filter
years = df["timestamp"].dt.year.unique()
year_filter = st.sidebar.multiselect("Select Year(s)", years, default=years)

# Apply filter
filtered_df = df[df["timestamp"].dt.year.isin(year_filter)]

# =========================
# Data Preview
# =========================
st.subheader("📊 Dataset Preview")
st.write(filtered_df.head())

st.markdown(f"**Total records:** {len(filtered_df)}")

# =========================
# Time-Series Chart
# =========================
st.subheader("📈 PM2.5 Over Time")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(filtered_df["timestamp"], filtered_df["PM2.5"], label="PM2.5", color="red")
ax.set_xlabel("Date")
ax.set_ylabel("PM2.5 Level")
ax.legend()
st.pyplot(fig)

# =========================
# Correlation Analysis
# =========================
st.subheader("📉 Correlation Heatmap")

numeric_cols = ["T", "TM", "Tm", "SLP", "H", "VV", "V", "VM", "PM2.5"]
corr = filtered_df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(corr, cmap="coolwarm")
fig.colorbar(cax)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels(numeric_cols, rotation=45, ha="left")
ax.set_yticklabels(numeric_cols)
st.pyplot(fig)

# =========================
# Summary Stats
# =========================
st.subheader("📑 Summary Statistics")
st.write(filtered_df.describe())
