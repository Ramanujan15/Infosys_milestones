import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def show_time_series(df, pollutants):
    """Plot line charts for selected pollutants."""
    st.subheader("📈 Time Series Plot")
    for pol in pollutants:
        st.line_chart(df.set_index("timestamp")[pol])

def show_correlation(df, pollutants):
    """Display correlation heatmap for pollutants."""
    st.subheader("🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df[pollutants].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

def show_summary(df, pollutants):
    """Show summary statistics for pollutants."""
    st.subheader("📑 Summary Statistics")
    st.write(df[pollutants].describe())
