import streamlit as st
import pandas as pd
import numpy as np
import os

from predict import predict, predict_top_k, LABELS

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HAR System",
    page_icon="🏃",
    layout="centered",
)

st.title("🏃 Human Activity Recognition")
st.caption("CNN-LSTM model trained on UCI HAR Dataset · 96.4% accuracy")
st.write("Upload sensor data (CSV: up to 128 rows × 9 columns)")
st.write("**Columns:** body_acc_x/y/z · body_gyro_x/y/z · total_acc_x/y/z")

# ── File uploader ─────────────────────────────────────────────────────────
file = st.file_uploader("Upload CSV", type=["csv"])

def show_result(data: np.ndarray):
    """Run prediction and display results."""
    label, conf = predict(data)
    top_k       = predict_top_k(data, k=3)

    # Main result
    col1, col2 = st.columns(2)
    col1.metric("Prediction", label)
    col2.metric("Confidence", f"{conf*100:.1f}%")

    # Top-3
    st.subheader("Top-3 Predictions")
    for rank, (lbl, c) in enumerate(top_k, 1):
        st.progress(c, text=f"{rank}. {lbl}  —  {c*100:.1f}%")

    # Signal chart
    st.subheader("Sensor Signals")
    df_plot = pd.DataFrame(
        data[:, :3],
        columns=["body_acc_x", "body_acc_y", "body_acc_z"]
    )
    st.line_chart(df_plot)

if file:
    try:
        df   = pd.read_csv(file)
        data = df.values.astype(np.float32)
        st.success(f"Loaded: {data.shape[0]} rows × {data.shape[1]} cols")
        show_result(data)
    except Exception as e:
        st.error(f"Error: {e}")


# ── Sidebar info ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write(
        "This app classifies human activities from raw inertial sensor data "
        "using a CNN-LSTM deep learning model."
    )
    st.subheader("Activity Classes")
    for lbl in LABELS:
        st.write(f"• {lbl}")
    st.subheader("Model")
    st.write("Architecture: CNN-LSTM (5 ConvBlocks + 2-layer LSTM)")
    st.write("Input: 128 × 9 raw sensor window")
    st.write("Dataset: UCI HAR (10,299 windows, 30 subjects)")
    st.write("Test accuracy: **96.40%**")
