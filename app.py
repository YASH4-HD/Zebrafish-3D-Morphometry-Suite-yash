import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

st.set_page_config(page_title="Zebrafish Morphometry Pro", layout="wide")

st.title("🧬 Cdh2-CRISPR Morphometry Suite")
st.markdown("*Advanced Spatial Analysis for Developmental Phenotyping*")

uploaded_file = st.sidebar.file_uploader("Upload CRISPR Dataset (CSV)", type="csv")

if uploaded_file:
    # 1. Robust Data Cleaning
    df = pd.read_csv(uploaded_file)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=['centroid-0', 'centroid-1', 'centroid-2'])

    # 2. NEW: Z-Axis Filtering (Slice Analysis)
    z_min, z_max = float(df['centroid-0'].min()), float(df['centroid-0'].max())
    z_range = st.sidebar.slider("Select Z-Depth Slice (μm)", z_min, z_max, (z_min, z_max))
    
    # Filter dataframe based on slider
    df_filtered = df[(df['centroid-0'] >= z_range[0]) & (df['centroid-0'] <= z_range[1])]

    # 3. Advanced Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Nuclei in Slice", len(df_filtered))
    
    # Calculate Correlation for Scientific Rigor
    if len(df_filtered) > 2:
        corr, _ = pearsonr(df_filtered['nearest_neighbor_dist'], df_filtered['volume_voxels'])
        m2.metric("Vol-NN Correlation", f"{corr:.2f}")
    
    m3.metric("Avg Volume", f"{df_filtered['volume_voxels'].mean():,.0f}")
    m4.metric("Packing Density (NN)", f"{df_filtered['nearest_neighbor_dist'].mean():.2f}")

    st.markdown("---")

    # 4. Visualization Layout
    col_left, col_right = st.columns([1.2, 1])

   with col_left:
    st.subheader("📍 3D Spatial Phenotype")
    fig3d = plt.figure(figsize=(7, 6), dpi=100)
    # ... (your existing plotting code for fig3d) ...
    st.pyplot(fig3d)
    
    # NEW: Download button for 3D Plot
    buf3d = io.BytesIO()
    fig3d.savefig(buf3d, format="png", dpi=300)
    st.download_button(
        label="📸 Download 3D Plot (PNG)",
        data=buf3d.getvalue(),
        file_name="3d_spatial_phenotype.png",
        mime="image/png"
    )

with col_right:
    st.subheader("📈 Quantitative Analysis")
    
    # Plot 1: Volume Distribution
    fig1, ax1 = plt.subplots(figsize=(6, 3), dpi=100)
    # ... (your existing plotting code for fig1) ...
    st.pyplot(fig1)
    
    # NEW: Download button for Volume Hist
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", dpi=300)
    st.download_button(
        label="📊 Download Volume Distribution",
        data=buf1.getvalue(),
        file_name="volume_distribution.png",
        mime="image/png"
    )
    
    # Plot 2: Spatial Packing
    fig2, ax2 = plt.subplots(figsize=(6, 3), dpi=100)
    # ... (your existing plotting code for fig2) ...
    st.pyplot(fig2)

    # NEW: Download button for Packing Plot
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=300)
    st.download_button(
        label="📉 Download Packing Analysis",
        data=buf2.getvalue(),
        file_name="packing_analysis.png",
        mime="image/png"
    )

    # 5. Data Export Section
    st.markdown("---")
    st.subheader("💾 Research Data Export")
    csv_data = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Analyzed Data as CSV",
        data=csv_data,
        file_name="cdh2_analyzed_results.csv",
        mime="text/csv",
    )

else:
    st.info("📂 Please upload the Cdh2-CRISPR dataset to begin analysis.")
