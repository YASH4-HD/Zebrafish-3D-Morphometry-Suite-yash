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
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    df = df.dropna(subset=['centroid-0', 'centroid-1', 'centroid-2'])

    # 2. Metrics
    st.subheader("📊 Quantitative Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Nuclei", len(df))
    m2.metric("Z-Span", f"{df['centroid-0'].max() - df['centroid-0'].min():.2f}")
    m3.metric("Avg Volume", f"{df['volume_voxels'].mean():,.0f}")

    # 3. 3D Plot (Plotly)
    st.subheader("📍 3D Spatial Distribution")
    # We use a very simple trace here to ensure it renders
    fig3d = go.Figure(data=[go.Scatter3d(
        x=df['centroid-2'],
        y=df['centroid-1'],
        z=df['centroid-0'],
        mode='markers',
        marker=dict(size=5, color='red', opacity=0.8)
    )])
    
    fig3d.update_layout(
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=0),
        height=500
    )
    # Using 'theme=None' to prevent Streamlit from messing with colors
    st.plotly_chart(fig3d, use_container_width=True, theme=None)

    # 4. 2D Plots (Using Matplotlib for guaranteed visibility)
    st.markdown("---")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📈 Depth Profile (Z)")
        fig, ax = plt.subplots()
        ax.hist(df['centroid-0'], bins=20, color='teal', edgecolor='black')
        ax.set_xlabel('Z-Coordinate')
        ax.set_ylabel('Count')
        st.pyplot(fig)
        
    with c2:
        st.subheader("🎯 XY Projection")
        fig, ax = plt.subplots()
        ax.scatter(df['centroid-2'], df['centroid-1'], c=df['centroid-0'], cmap='viridis', s=20)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        st.pyplot(fig)

    with st.expander("🔍 Raw Data Table"):
        st.dataframe(df)
else:
    st.info("Please upload your CSV file.")
