import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set page to wide to allow side-by-side charts
st.set_page_config(page_title="Zebrafish 3D Morphometry", layout="wide")

st.title("🔬 Zebrafish 3D Morphometry Suite")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    # 1. Data Processing
    df = pd.read_csv(uploaded_file)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df = df.dropna(subset=['centroid-0', 'centroid-1', 'centroid-2'])

    # 2. Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Nuclei", len(df))
    m2.metric("Z-Span", f"{df['centroid-0'].max() - df['centroid-0'].min():.2f}")
    m3.metric("Avg Volume", f"{df['volume_voxels'].mean():,.0f}")
    m4.metric("Avg NN Dist", f"{df['nearest_neighbor_dist'].mean():.2f}")

    st.markdown("---")

    # 3. Side-by-Side Layout for Charts
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📍 3D Spatial Distribution")
        # Reduced figsize from (10,7) to (6,5) for a compact look
        fig3d = plt.figure(figsize=(6, 5), dpi=100)
        ax3d = fig3d.add_subplot(111, projection='3d')
        
        p = ax3d.scatter(df['centroid-2'], df['centroid-1'], df['centroid-0'], 
                         c=df['centroid-0'], cmap='viridis', s=15, alpha=0.6)
        
        ax3d.set_xlabel('X', fontsize=8)
        ax3d.set_ylabel('Y', fontsize=8)
        ax3d.set_zlabel('Z', fontsize=8)
        # Clean up the 3D panes for a modern look
        ax3d.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax3d.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        plt.colorbar(p, ax=ax3d, label='Depth', shrink=0.5, pad=0.1)
        st.pyplot(fig3d)

    with col_right:
        st.subheader("📈 Morphometric Analysis")
        
        # Top 2D plot: Depth Histogram
        fig1, ax1 = plt.subplots(figsize=(6, 2.5), dpi=100) # Short height
        ax1.hist(df['centroid-0'], bins=30, color='#2E8B57', edgecolor='white', alpha=0.8)
        ax1.set_title('Depth Profile (Z)', fontsize=10)
        ax1.tick_params(labelsize=8)
        st.pyplot(fig1)
        
        # Bottom 2D plot: Volume vs NN
        fig2, ax2 = plt.subplots(figsize=(6, 2.5), dpi=100) # Short height
        ax2.scatter(df['nearest_neighbor_dist'], df['volume_voxels'], 
                    c=df['centroid-0'], cmap='plasma', s=10, alpha=0.5)
        ax2.set_title('Volume vs Neighborhood Density', fontsize=10)
        ax2.set_xlabel('NN Dist', fontsize=8)
        ax2.set_ylabel('Volume', fontsize=8)
        ax2.tick_params(labelsize=8)
        st.pyplot(fig2)

    # 4. Data Inspector
    with st.expander("🔍 View Raw Data Table"):
        st.dataframe(df, use_container_width=True)

else:
    st.info("👋 Analysis Ready. Please upload your CSV file.")
