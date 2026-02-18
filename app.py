import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Zebrafish 3D Morphometry", layout="wide")

st.title("🔬 Zebrafish 3D Morphometry Suite")
st.markdown("Comprehensive analysis of 3D nuclei segmentation outputs.")

# =========================================================
# Utilities
# =========================================================

def ensure_numeric(df, columns):
    out = df.copy()
    for col in columns:
        if col in out.columns:
            # Handle commas in large numbers (e.g. 596,512 -> 596512)
            out[col] = out[col].astype(str).str.replace(",", "", regex=False)
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def compute_knn_metrics(df, x_col, y_col, z_col, k=5):
    points = df[[x_col, y_col, z_col]].to_numpy(float)
    n = len(points)
    if n < 3: return pd.DataFrame(index=df.index)
    diff = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    k_eff = max(1, min(k, n - 1))
    nn_idx = np.argpartition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]
    nn_dists = np.take_along_axis(dists, nn_idx, axis=1)
    return pd.DataFrame({"nn_mean_dist": nn_dists.mean(axis=1), "nn_degree": k_eff}, index=df.index)

# =========================================================
# Data Input & Processing
# =========================================================

st.sidebar.header("📂 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload nuclei CSV file", type="csv")

if not uploaded_file:
    st.info("👋 Please upload your segmentation CSV file to begin.")
    st.stop()

raw_df = pd.read_csv(uploaded_file)
cols = {c.lower(): c for c in raw_df.columns}

# Map columns
z_col, y_col, x_col = cols.get("centroid-0"), cols.get("centroid-1"), cols.get("centroid-2")
vol_col = cols.get("volume_voxels")

if not (x_col and y_col and z_col):
    st.error("Centroid columns not found.")
    st.stop()

df = ensure_numeric(raw_df, [x_col, y_col, z_col, vol_col] if vol_col else [x_col, y_col, z_col])
df = df.dropna(subset=[x_col, y_col, z_col]).copy()

# Add KNN and PCA
knn = compute_knn_metrics(df, x_col, y_col, z_col)
df = df.join(knn)

coords = df[[x_col, y_col, z_col]].to_numpy(float)
centered = coords - coords.mean(axis=0)
cov = np.cov(centered, rowvar=False)
evals, evecs = np.linalg.eigh(cov)
order = np.argsort(evals)[::-1]
evecs = evecs[:, order]
pc1 = centered @ evecs[:, 0]
df["pseudotime"] = (pc1 - pc1.min()) / (pc1.max() - pc1.min())
anisotropy_ratio = float(evals.max() / max(evals.min(), 1e-9))

# Radial distance
lx, ly, lz = df[[x_col, y_col, z_col]].mean()
df["radial_distance"] = np.sqrt((df[x_col]-lx)**2 + (df[y_col]-ly)**2 + (df[z_col]-lz)**2)

# =========================================================
# Visualizations with FORCE-ZOOM
# =========================================================

st.subheader("📊 Quantitative Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Total Nuclei", len(df))
c2.metric("Z-Depth Span", f"{(df[z_col].max() - df[z_col].min()):.2f}")
c3.metric("Anisotropy Ratio", f"{anisotropy_ratio:.2f}")

# Calculate global limits for charts
x_range = [df[x_col].min()*0.9, df[x_col].max()*1.1]
y_range = [df[y_col].min()*0.9, df[y_col].max()*1.1]
z_range = [df[z_col].min()*0.9, df[z_col].max()*1.1]

st.subheader("📍 3D Spatial Distribution")
fig3d = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color="nn_mean_dist", 
                     color_continuous_scale="Viridis", height=700)
# Force 3D camera to focus on data
fig3d.update_layout(scene=dict(xaxis=dict(range=x_range), yaxis=dict(range=y_range), zaxis=dict(range=z_range)))
st.plotly_chart(fig3d, use_container_width=True)

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📈 Depth Profile")
    fig_depth = px.histogram(df, x=z_col, nbins=25, template="plotly_white")
    fig_depth.update_xaxes(range=[df[z_col].min(), df[z_col].max()])
    st.plotly_chart(fig_depth, use_container_width=True)

with col_b:
    st.subheader("🎯 Radial Morphometry")
    fig_radial = px.scatter(df, x="radial_distance", y="nn_mean_dist", template="plotly_white")
    # Force zoom to data
    fig_radial.update_xaxes(range=[df["radial_distance"].min(), df["radial_distance"].max()])
    fig_radial.update_yaxes(range=[df["nn_mean_dist"].min(), df["nn_mean_dist"].max()])
    st.plotly_chart(fig_radial, use_container_width=True)

with st.expander("🔍 Inspect Processed Data"):
    st.dataframe(df, use_container_width=True)
