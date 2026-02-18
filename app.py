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
            out[col] = (
                out[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.strip()
                .replace({"": np.nan, "nan": np.nan, "None": np.nan})
            )
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def compute_knn_metrics(df, x_col, y_col, z_col, k=5):
    points = df[[x_col, y_col, z_col]].to_numpy(float)
    n = len(points)

    if n < 3:
        return pd.DataFrame(index=df.index)

    diff = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)

    k_eff = max(1, min(k, n - 1))
    nn_idx = np.argpartition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]
    nn_dists = np.take_along_axis(dists, nn_idx, axis=1)

    return pd.DataFrame({
        "nn_mean_dist": nn_dists.mean(axis=1),
        "nn_degree": k_eff
    }, index=df.index)


# =========================================================
# Data Input
# =========================================================

st.sidebar.header("📂 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload nuclei CSV file", type="csv")

if not uploaded_file:
    st.info("👋 Please upload your segmentation CSV file to begin.")
    st.stop()

raw_df = pd.read_csv(uploaded_file)
cols = {c.lower(): c for c in raw_df.columns}

z_col = cols.get("centroid-0")
y_col = cols.get("centroid-1")
x_col = cols.get("centroid-2")
vol_col = cols.get("volume_voxels")
nn_col = cols.get("nearest_neighbor_dist")

if not (x_col and y_col and z_col):
    st.error("Centroid columns not found.")
    st.stop()

numeric_cols = [x_col, y_col, z_col]
if vol_col:
    numeric_cols.append(vol_col)
if nn_col:
    numeric_cols.append(nn_col)

df = ensure_numeric(raw_df, numeric_cols)
df = df.dropna(subset=[x_col, y_col, z_col]).copy()

if len(df) < 3:
    st.error("Not enough valid rows.")
    st.stop()


# =========================================================
# KNN + PCA
# =========================================================

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


# =========================================================
# Radial Distance (SAFE)
# =========================================================

lx, ly, lz = df[[x_col, y_col, z_col]].mean()

df["radial_distance"] = np.sqrt(
    (df[x_col] - lx) ** 2 +
    (df[y_col] - ly) ** 2 +
    (df[z_col] - lz) ** 2
)

df["radial_distance"] = pd.to_numeric(df["radial_distance"], errors="coerce")
df["nn_mean_dist"] = pd.to_numeric(df["nn_mean_dist"], errors="coerce")

df = df.dropna(subset=["radial_distance", "nn_mean_dist"])


# =========================================================
# Summary
# =========================================================

st.subheader("📊 Quantitative Summary")
c1, c2, c3 = st.columns(3)

c1.metric("Total Nuclei", len(df))
c2.metric("Z-Depth Span", f"{(df[z_col].max() - df[z_col].min()):.2f}")
c3.metric("Anisotropy Ratio", f"{anisotropy_ratio:.2f}")


# =========================================================
# 3D Plot
# =========================================================

st.subheader("📍 3D Spatial Distribution")

fig3d = px.scatter_3d(
    df,
    x=x_col,
    y=y_col,
    z=z_col,
    template="plotly_white",
    height=600
)

fig3d.update_traces(marker=dict(size=4))
st.plotly_chart(fig3d, use_container_width=True)


# =========================================================
# Depth Histogram
# =========================================================

st.subheader("📈 Depth Profile")

fig_depth = px.histogram(
    df,
    x=z_col,
    nbins=25,
    template="plotly_white",
    height=500
)

st.plotly_chart(fig_depth, use_container_width=True)


# =========================================================
# Volume Histogram
# =========================================================

if vol_col:
    st.subheader("📊 Volume Distribution")

    fig_vol = px.histogram(
        df,
        x=vol_col,
        nbins=25,
        template="plotly_white",
        height=500
    )

    fig_vol.update_xaxes(type="log")
    st.plotly_chart(fig_vol, use_container_width=True)


# =========================================================
# Radial Morphometry
# =========================================================

st.subheader("🎯 Radial Morphometry")

fig_radial = px.scatter(
    df,
    x="radial_distance",
    y="nn_mean_dist",
    template="plotly_white",
    height=550
)

st.plotly_chart(fig_radial, use_container_width=True)


# =========================================================
# Pseudotime Trend
# =========================================================

st.subheader("⏱️ Temporal Trajectory")

trend_df = df.copy()
trend_df["pt_bin"] = pd.cut(trend_df["pseudotime"], bins=10)

trend = trend_df.groupby("pt_bin", observed=False)[["nn_mean_dist"]].mean().reset_index()
trend["pt_bin"] = trend["pt_bin"].astype(str)

fig_trend = px.line(
    trend,
    x="pt_bin",
    y="nn_mean_dist",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig_trend, use_container_width=True)


# =========================================================
# Inspect Data
# =========================================================

with st.expander("🔍 Inspect Processed Data"):
    st.dataframe(df, use_container_width=True)
