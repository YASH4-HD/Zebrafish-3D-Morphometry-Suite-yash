import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Zebrafish 3D Morphometry", layout="wide")

st.title("🔬 Zebrafish 3D Morphometry Suite")
st.markdown("Comprehensive analysis of 3D nuclei segmentation outputs.")

# =========================================================
# 1. Data Cleaning (Crucial for your specific CSV)
# =========================================================
def clean_bio_data(df):
    out = df.copy()
    for col in out.columns:
        # Remove commas from numbers like "596,512"
        if out[col].dtype == 'object':
            out[col] = out[col].astype(str).str.replace(',', '', regex=False)
        out[col] = pd.to_numeric(out[col], errors='coerce')
    return out

# =========================================================
# 2. KNN & PCA Logic
# =========================================================
def compute_metrics(df, x, y, z):
    pts = df[[x, y, z]].to_numpy(float)
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    nn_dist = np.sort(dists, axis=1)[:, :5].mean(axis=1)
    
    # PCA for Pseudotime
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    pc1 = centered @ evecs[:, np.argsort(evals)[-1]]
    pseudotime = (pc1 - pc1.min()) / (pc1.max() - pc1.min())
    
    return nn_dist, pseudotime, float(evals.max() / max(evals.min(), 1e-9))

# =========================================================
# 3. Main App Execution
# =========================================================
st.sidebar.header("📂 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload nuclei CSV file", type="csv")

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    df = clean_bio_data(raw_df)
    
    # Column mapping
    z_col, y_col, x_col = 'centroid-0', 'centroid-1', 'centroid-2'
    vol_col = 'volume_voxels'
    
    df = df.dropna(subset=[x_col, y_col, z_col])
    
    # Run Calculations
    nn_dist, ptime, anisotropy = compute_metrics(df, x_col, y_col, z_col)
    df['nn_mean_dist'] = nn_dist
    df['pseudotime'] = ptime
    
    # Radial Distance
    lx, ly, lz = df[x_col].mean(), df[y_col].mean(), df[z_col].mean()
    df['radial_distance'] = np.sqrt((df[x_col]-lx)**2 + (df[y_col]-ly)**2 + (df[z_col]-lz)**2)

    # Metrics Row
    st.subheader("📊 Quantitative Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Nuclei", len(df))
    c2.metric("Z-Depth Span", f"{(df[z_col].max() - df[z_col].min()):.2f}")
    c3.metric("Anisotropy Ratio", f"{anisotropy:.2f}")

    # 3D Plot with MANUAL SCALE
    st.subheader("📍 3D Spatial Distribution")
    fig3d = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color='nn_mean_dist', 
                         color_continuous_scale='Viridis', height=600)
    # FORCE ZOOM
    fig3d.update_layout(scene=dict(
        xaxis=dict(range=[df[x_col].min(), df[x_col].max()]),
        yaxis=dict(range=[df[y_col].min(), df[y_col].max()]),
        zaxis=dict(range=[df[z_col].min(), df[z_col].max()])
    ))
    st.plotly_chart(fig3d, use_container_width=True)

    # 2D Plots
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📈 Depth Profile")
        fig_h = px.histogram(df, x=z_col, nbins=20, color_discrete_sequence=['#00CC96'])
        fig_h.update_xaxes(range=[df[z_col].min(), df[z_col].max()])
        st.plotly_chart(fig_h, use_container_width=True)
    with col_b:
        st.subheader("🎯 Radial Morphometry")
        fig_r = px.scatter(df, x='radial_distance', y='nn_mean_dist', color='pseudotime')
        fig_r.update_xaxes(range=[df['radial_distance'].min(), df['radial_distance'].max()])
        st.plotly_chart(fig_r, use_container_width=True)

    with st.expander("🔍 Inspect Processed Data"):
        st.dataframe(df)
else:
    st.info("Please upload your CSV to begin.")
