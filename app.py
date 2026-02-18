import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="Zebrafish 3D Morphometry", layout="wide")

st.title("🔬 Zebrafish 3D Morphometry Suite")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    # 1. Load and Force Numeric
    df = pd.read_csv(uploaded_file)
    for col in df.columns:
        # Clean commas from volume and convert to float
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    df = df.dropna(subset=['centroid-0', 'centroid-1', 'centroid-2'])

    # 2. Metrics (Top Row)
    st.subheader("📊 Quantitative Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Nuclei", len(df))
    m2.metric("Z-Span", f"{df['centroid-0'].max() - df['centroid-0'].min():.2f}")
    m3.metric("Avg Volume", f"{df['volume_voxels'].mean():,.0f}")
    m4.metric("Avg NN Dist", f"{df['nearest_neighbor_dist'].mean():.2f}")

    # 3. 3D Spatial Plot (Force WebGL Rendering)
    st.subheader("📍 3D Spatial Distribution")
    
    fig3d = go.Figure(data=[go.Scatter3d(
        x=df['centroid-2'],
        y=df['centroid-1'],
        z=df['centroid-0'],
        mode='markers',
        marker=dict(
            size=3,
            color=df['centroid-0'], 
            colorscale='Viridis',
            opacity=0.7
        )
    )])
    
    fig3d.update_layout(
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    
    # FORCE RENDERER and unique key
    st.plotly_chart(fig3d, use_container_width=True, theme=None, key="final_3d_render")

    # 4. 2D Distribution Plots (Using the Matplotlib logic that worked)
    st.markdown("---")
    st.subheader("🧬 Morphometric Analysis")
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("### 📈 Depth Profile (Z)")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.hist(df['centroid-0'], bins=25, color='#2E8B57', edgecolor='white')
        ax1.set_xlabel('Z-Coordinate (Depth)')
        ax1.set_ylabel('Count')
        st.pyplot(fig1)
        
    with c2:
        st.markdown("### 🎯 Volume vs Neighborhood Density")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.scatter(df['nearest_neighbor_dist'], df['volume_voxels'], c=df['centroid-0'], cmap='plasma', alpha=0.6)
        ax2.set_xlabel('NN Distance')
        ax2.set_ylabel('Volume (voxels)')
        st.pyplot(fig2)

    # 5. Data Inspector
    with st.expander("🔍 Inspect Processed Data Table"):
        st.dataframe(df, use_container_width=True)
else:
    st.info("👋 Analysis Ready. Please upload your CSV file.")
