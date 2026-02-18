import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Zebrafish 3D Morphometry", layout="wide")

st.title("🔬 Zebrafish 3D Morphometry Suite")

# 1. Sidebar & File Upload
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Clean commas and force numbers
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['centroid-0', 'centroid-1', 'centroid-2'])

    # 2. Metrics
    st.subheader("📊 Quantitative Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Nuclei", len(df))
    m2.metric("Z-Span", f"{df['centroid-0'].max() - df['centroid-0'].min():.2f}")
    m3.metric("Avg Volume", f"{df['volume_voxels'].mean():,.0f}")

    # 3. 3D Spatial Plot (Using Graph Objects for better control)
    st.subheader("📍 3D Spatial Distribution")
    fig3d = go.Figure(data=[go.Scatter3d(
        x=df['centroid-2'],
        y=df['centroid-1'],
        z=df['centroid-0'],
        mode='markers',
        marker=dict(size=4, color=df['centroid-0'], colorscale='Viridis', opacity=0.8)
    )])
    
    # MANUALLY FORCE THE BOX SIZE
    fig3d.update_layout(
        scene=dict(
            xaxis=dict(range=[df['centroid-2'].min(), df['centroid-2'].max()], title='X'),
            yaxis=dict(range=[df['centroid-1'].min(), df['centroid-1'].max()], title='Y'),
            zaxis=dict(range=[df['centroid-0'].min(), df['centroid-0'].max()], title='Z'),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # 4. 2D Distribution Plots
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📈 Depth Profile")
        fig_z = go.Figure(data=[go.Histogram(x=df['centroid-0'], nbinsx=20, marker_color='#00CC96')])
        fig_z.update_layout(xaxis=dict(range=[df['centroid-0'].min(), df['centroid-0'].max()]), template="plotly_white")
        st.plotly_chart(fig_z, use_container_width=True)
        
    with c2:
        st.subheader("🎯 XY Density")
        fig_xy = go.Figure(data=[go.Scatter(
            x=df['centroid-2'], y=df['centroid-1'], mode='markers',
            marker=dict(color=df['centroid-0'], colorscale='Plasma')
        )])
        fig_xy.update_layout(
            xaxis=dict(range=[df['centroid-2'].min(), df['centroid-2'].max()]),
            yaxis=dict(range=[df['centroid-1'].min(), df['centroid-1'].max()]),
            template="plotly_white"
        )
        st.plotly_chart(fig_xy, use_container_width=True)

    with st.expander("🔍 Raw Data Table"):
        st.dataframe(df)

else:
    st.info("Please upload your CSV file.")
