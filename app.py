import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Zebrafish 3D Morphometry", layout="wide")

st.title("🔬 Zebrafish 3D Morphometry Suite")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    # 1. Load and Force Numeric (Handling those commas in volume)
    df = pd.read_csv(uploaded_file)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    df = df.dropna(subset=['centroid-0', 'centroid-1', 'centroid-2'])

    # 2. Metrics (Top Row)
    st.subheader("📊 Quantitative Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Nuclei", len(df))
    m2.metric("Z-Span", f"{df['centroid-0'].max() - df['centroid-0'].min():.2f}")
    m3.metric("Avg Volume", f"{df['volume_voxels'].mean():,.0f}")
    m4.metric("Avg NN Dist", f"{df['nearest_neighbor_dist'].mean():.2f}")

    # 3. 3D Spatial Plot (With forced rendering settings)
    st.subheader("📍 3D Spatial Distribution")
    
    fig3d = go.Figure(data=[go.Scatter3d(
        x=df['centroid-2'],
        y=df['centroid-1'],
        z=df['centroid-0'],
        mode='markers',
        marker=dict(
            size=4,
            color=df['centroid-0'], # Color by depth
            colorscale='Viridis',
            opacity=0.8,
            showscale=True
        )
    )])
    
    fig3d.update_layout(
        scene=dict(
            xaxis_title='X (Centroid-2)',
            yaxis_title='Y (Centroid-1)',
            zaxis_title='Z (Centroid-0)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600,
        paper_bgcolor='rgba(0,0,0,0)', # Transparent background
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Use theme=None and specify a unique key to force a fresh render
    st.plotly_chart(fig3d, use_container_width=True, theme=None, key="zebrafish_3d_final")

    # 4. 2D Distribution Plots (Matplotlib - Proven to work)
    st.markdown("---")
    st.subheader("🧬 Morphometric Analysis")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### 📈 Depth Profile (Z)")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.hist(df['centroid-0'], bins=25, color='#008080', edgecolor='white', alpha=0.8)
        ax1.set_xlabel('Z-Coordinate (Depth)')
        ax1.set_ylabel('Nuclei Count')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(fig1)
        
    with c2:
        st.markdown("### 🎯 XY Spatial Projection")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        scatter = ax2.scatter(df['centroid-2'], df['centroid-1'], c=df['centroid-0'], cmap='viridis', s=30, alpha=0.7)
        ax2.set_xlabel('X (Centroid-2)')
        ax2.set_ylabel('Y (Centroid-1)')
        plt.colorbar(scatter, ax=ax2, label='Depth (Z)')
        st.pyplot(fig2)

    # 5. Data Inspector
    with st.expander("🔍 Inspect Processed Data Table"):
        st.dataframe(df, use_container_width=True)
else:
    st.info("👋 Welcome! Please upload your 'zebrafish_nuclei_data.csv' to begin the analysis.")
