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
        ax3d = fig3d.add_subplot(111, projection='3d')
        
        # Color by Volume to show phenotypic heterogeneity
        p = ax3d.scatter(df_filtered['centroid-2'], df_filtered['centroid-1'], df_filtered['centroid-0'], 
                         c=df_filtered['volume_voxels'], cmap='magma', s=20, alpha=0.7)
        
        ax3d.set_xlabel('X (Centroid-2)', fontsize=9, labelpad=10)
        ax3d.set_ylabel('Y (Centroid-1)', fontsize=9, labelpad=10)
        ax3d.set_zlabel('Z (Depth)', fontsize=9, labelpad=10)
        
        # Modern Styling
        ax3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        plt.colorbar(p, ax=ax3d, label='Nucleus Volume (voxels)', shrink=0.5, pad=0.15)
        fig3d.tight_layout()
        st.pyplot(fig3d)

    with col_right:
        st.subheader("📈 Quantitative Analysis")
        
        # Plot 1: Volume Distribution
        fig1, ax1 = plt.subplots(figsize=(6, 3), dpi=100)
        ax1.hist(df_filtered['volume_voxels'], bins=25, color='#8A2BE2', edgecolor='white', alpha=0.7)
        ax1.set_title('Volumetric Heterogeneity', fontsize=10)
        ax1.set_xlabel('Volume (voxels)', fontsize=8)
        fig1.tight_layout()
        st.pyplot(fig1)
        
        # Plot 2: Spatial Packing (NN Dist vs Z-Depth)
        fig2, ax2 = plt.subplots(figsize=(6, 3), dpi=100)
        ax2.scatter(df_filtered['centroid-0'], df_filtered['nearest_neighbor_dist'], 
                    c=df_filtered['nearest_neighbor_dist'], cmap='viridis', s=15, alpha=0.6)
        ax2.set_title('Packing Density vs. Tissue Depth', fontsize=10)
        ax2.set_xlabel('Z-Depth', fontsize=8)
        ax2.set_ylabel('NN Distance', fontsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)

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
