import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Zebrafish 3D Morphometry", layout="wide")

st.title("🔬 Zebrafish 3D Morphometry Suite")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    # 1. Load data
    df = pd.read_csv(uploaded_file)
    
    # 2. AGGRESSIVE CLEANING: Force everything to numbers
    for col in df.columns:
        # Remove commas and whitespace, then convert to numeric
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    # Drop rows that have missing coordinates
    df = df.dropna(subset=['centroid-0', 'centroid-1', 'centroid-2'])

    # 3. Metrics
    st.subheader("📊 Quantitative Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Nuclei", len(df))
    m2.metric("Z-Span", f"{df['centroid-0'].max() - df['centroid-0'].min():.2f}")
    m3.metric("Avg Volume", f"{df['volume_voxels'].mean():,.0f}")

    # 4. 3D Plot
    st.subheader("📍 3D Spatial Distribution")
    fig3d = go.Figure(data=[go.Scatter3d(
        x=df['centroid-2'],
        y=df['centroid-1'],
        z=df['centroid-0'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['centroid-0'], 
            colorscale='Viridis',
            opacity=0.9
        )
    )])
    
    fig3d.update_layout(
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    st.plotly_chart(fig3d, use_container_width=True)

    # 5. 2D Plots
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📈 Depth Profile")
        # Explicitly setting the histogram data
        fig_z = go.Figure(data=[go.Histogram(
            x=df['centroid-0'], 
            nbinsx=25,
            marker_color='#00CC96'
        )])
        fig_z.update_layout(template="plotly_white", xaxis_title="Z-Coordinate")
        st.plotly_chart(fig_z, use_container_width=True)
        
    with c2:
        st.subheader("🎯 XY Density")
        fig_xy = go.Figure(data=[go.Scatter(
            x=df['centroid-2'], 
            y=df['centroid-1'], 
            mode='markers',
            marker=dict(
                color=df['centroid-0'],
                colorscale='Plasma',
                size=8
            )
        )])
        fig_xy.update_layout(template="plotly_white", xaxis_title="X", yaxis_title="Y")
        st.plotly_chart(fig_xy, use_container_width=True)

    with st.expander("🔍 Raw Data Table"):
        st.dataframe(df)

else:
    st.info("Please upload the 'zebrafish_nuclei_data.csv' file to start.")
