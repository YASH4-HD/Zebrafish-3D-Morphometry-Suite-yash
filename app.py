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


def ensure_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Force numeric parsing, tolerating comma-formatted values."""
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


def compute_knn_metrics(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, k: int = 5):
    points = df[[x_col, y_col, z_col]].to_numpy(dtype=float)
    n = len(points)
    if n < 3:
        return pd.DataFrame(index=df.index, data={"nn_mean_dist": np.nan, "nn_degree": 0, "nn_clustering": np.nan})

    diff = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)

    k_eff = max(1, min(k, n - 1))
    nn_idx = np.argpartition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]
    nn_dists = np.take_along_axis(dists, nn_idx, axis=1)

    clustering = []
    for i in range(n):
        neigh = nn_idx[i]
        if len(neigh) < 2:
            clustering.append(0.0)
            continue
        local = dists[np.ix_(neigh, neigh)]
        radius = float(np.nanmedian(nn_dists[i]) * 1.2)
        connected = int(np.sum((local < radius) & np.isfinite(local)))
        possible = len(neigh) * (len(neigh) - 1)
        clustering.append(connected / possible if possible else 0.0)

    return pd.DataFrame(
        {
            "nn_mean_dist": nn_dists.mean(axis=1),
            "nn_degree": k_eff,
            "nn_clustering": clustering,
        },
        index=df.index,
    )


def simple_dbscan(points: np.ndarray, eps: float, min_pts: int):
    n = len(points)
    labels = np.full(n, -1, dtype=int)
    if n == 0:
        return labels

    visited = np.zeros(n, dtype=bool)
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.where(dists[i] <= eps)[0]
        if len(neighbors) < min_pts:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seed = set(neighbors.tolist())
        seed.discard(i)

        while seed:
            j = seed.pop()
            if not visited[j]:
                visited[j] = True
                jn = np.where(dists[j] <= eps)[0]
                if len(jn) >= min_pts:
                    seed.update(jn.tolist())
            if labels[j] < 0:
                labels[j] = cluster_id

        cluster_id += 1

    return labels


def assign_anatomical_regions(df: pd.DataFrame, x_col: str, y_col: str, z_col: str):
    out = df.copy()
    x_mid = out[x_col].median()
    y_mid = out[y_col].median()
    z_q1, z_q2 = out[z_col].quantile([0.33, 0.66])

    out["axis_dv"] = np.where(out[y_col] >= y_mid, "dorsal", "ventral")
    out["axis_lr"] = np.where(out[x_col] >= x_mid, "right", "left")
    out["axis_ap"] = pd.cut(
        out[z_col],
        bins=[-np.inf, z_q1, z_q2, np.inf],
        labels=["anterior", "mid", "posterior"],
    ).astype(str)
    out["region_label"] = out["axis_ap"] + "-" + out["axis_dv"] + "-" + out["axis_lr"]
    return out


def qc_report(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, vol_col: str | None, nn_col: str | None):
    report = [
        ("Duplicate centroids", int(df.duplicated(subset=[x_col, y_col, z_col]).sum()), "warn"),
        ("Missing centroid values", int(df[[x_col, y_col, z_col]].isna().sum().sum()), "fail"),
    ]
    if vol_col and vol_col in df.columns:
        report.append(("Non-positive volume", int((df[vol_col] <= 0).sum()), "fail"))
        report.append(("Extreme high volume", int((df[vol_col] > df[vol_col].quantile(0.995)).sum()), "warn"))
    if nn_col and nn_col in df.columns:
        report.append(("Negative NN distance", int((df[nn_col] < 0).sum()), "fail"))

    qc_df = pd.DataFrame(report, columns=["check", "count", "severity"])
    qc_df["status"] = np.where(qc_df["count"] == 0, "pass", qc_df["severity"])
    return qc_df[["check", "count", "status"]]


def make_methods_text(settings: dict):
    lines = [
        "### Auto-generated Methods Summary",
        "Nuclei coordinates were analyzed in the Zebrafish 3D Morphometry Suite.",
        f"Input centroid columns: x={settings['x_col']}, y={settings['y_col']}, z={settings['z_col']}.",
        f"Outlier filtering mode: {settings['filter_mode']}.",
        f"k-nearest-neighbor setting: k={settings['k_neighbors']}.",
        f"Cluster detection parameters: eps={settings['eps']:.2f}, min_pts={settings['min_pts']}.",
        "Anatomical regions were assigned by axis-wise quantile/median partitioning.",
        "Pseudo-time was estimated via the first PCA axis from 3D coordinates.",
        "Quality-control checks included duplicate centroid detection, null coordinates, and volume outliers.",
    ]
    return "\n\n".join(lines)


st.sidebar.header("📂 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload nuclei CSV file", type="csv")
comparison_files = st.sidebar.file_uploader(
    "Optional: Upload comparison/stage CSVs", type="csv", accept_multiple_files=True
)

st.sidebar.header("⚙️ Analysis Settings")
filter_mode = st.sidebar.selectbox("Outlier filtering", ["None", "Volume top 1%", "Volume top 5%"])
k_neighbors = st.sidebar.slider("k for neighborhood graph", 2, 15, 5)

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
    st.error("⚠️ Required centroid columns not found in CSV.")
    st.stop()

numeric_cols = [x_col, y_col, z_col] + ([vol_col] if vol_col else []) + ([nn_col] if nn_col else [])
df = ensure_numeric(raw_df, numeric_cols)
df = df.dropna(subset=[x_col, y_col, z_col]).copy()

if len(df) == 0:
    st.error("⚠️ No valid numeric centroid rows after parsing. Please check CSV formatting.")
    st.stop()

with st.sidebar.expander("🧪 Synthetic data stress testing"):
    enable_synth = st.checkbox("Enable synthetic augmentation", value=False)
    synth_n = st.slider("Synthetic nuclei count", 50, 800, 120, 10)
    synth_noise = st.slider("Coordinate noise", 0.0, 60.0, 12.0, 1.0)

if enable_synth:
    rng = np.random.default_rng(42)
    center = df[[x_col, y_col, z_col]].median().to_numpy()
    synth = rng.normal(loc=center, scale=synth_noise, size=(synth_n, 3))
    sdf = pd.DataFrame(synth, columns=[x_col, y_col, z_col])
    sdf["label"] = [f"synth_{i}" for i in range(synth_n)]
    if vol_col:
        sdf[vol_col] = rng.lognormal(mean=np.log(max(df[vol_col].median(), 1.0)), sigma=0.35, size=synth_n)
    if nn_col:
        sdf[nn_col] = np.nan
    df["is_synthetic"] = False
    sdf["is_synthetic"] = True
    df = pd.concat([df, sdf], ignore_index=True, sort=False)

if vol_col and filter_mode != "None":
    q = 0.99 if filter_mode == "Volume top 1%" else 0.95
    df = df[df[vol_col] <= df[vol_col].quantile(q)].copy()

if len(df) < 3:
    st.error("⚠️ Not enough rows for spatial analysis after filtering.")
    st.stop()

df = assign_anatomical_regions(df, x_col, y_col, z_col)

knn = compute_knn_metrics(df, x_col, y_col, z_col, k=k_neighbors)
for c in knn.columns:
    df[c] = knn[c]

eps_default = float(np.nanmedian(df["nn_mean_dist"]) * 1.5)
st.sidebar.subheader("🧩 Cluster detection")
eps = st.sidebar.slider("DBSCAN eps", 5.0, 200.0, float(np.clip(eps_default, 5.0, 200.0)), 1.0)
min_pts = st.sidebar.slider("DBSCAN min points", 3, 20, 6)
df["cluster_id"] = simple_dbscan(df[[x_col, y_col, z_col]].to_numpy(float), eps, min_pts)

coords = df[[x_col, y_col, z_col]].to_numpy(float)
centered = coords - coords.mean(axis=0)
cov = np.cov(centered, rowvar=False)
evals, evecs = np.linalg.eigh(cov)
order = np.argsort(evals)[::-1]
evals = evals[order]
evecs = evecs[:, order]
anisotropy_ratio = float(evals[0] / max(evals[-1], 1e-9))

pc1 = centered @ evecs[:, 0]
df["pseudotime"] = (pc1 - pc1.min()) / max(pc1.max() - pc1.min(), 1e-9)

if vol_col:
    vol_z = (df[vol_col] - df[vol_col].mean()) / max(df[vol_col].std(ddof=0), 1e-9)
else:
    vol_z = pd.Series(np.zeros(len(df)), index=df.index)
knn_z = (df["nn_mean_dist"] - df["nn_mean_dist"].mean()) / max(df["nn_mean_dist"].std(ddof=0), 1e-9)
df["outlier_score"] = vol_z.abs() + knn_z.abs()

qc_df = qc_report(df, x_col, y_col, z_col, vol_col, nn_col)

recipe = {
    "x_col": x_col,
    "y_col": y_col,
    "z_col": z_col,
    "filter_mode": filter_mode,
    "k_neighbors": int(k_neighbors),
    "eps": float(eps),
    "min_pts": int(min_pts),
}
st.sidebar.download_button(
    "💾 Download analysis recipe",
    data=json.dumps(recipe, indent=2),
    file_name="analysis_recipe.json",
    mime="application/json",
)

st.subheader("📊 Quantitative Summary")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Nuclei", len(df))
if vol_col:
    m2.metric("Avg Volume (voxels)", f"{df[vol_col].mean():,.1f}")
m3.metric("Z-Depth Span", f"{(df[z_col].max() - df[z_col].min()):.1f}")
if nn_col and nn_col in df.columns:
    m4.metric("Mean NN Distance", f"{df[nn_col].mean():.2f}")
else:
    m4.metric("Mean NN Distance", f"{df['nn_mean_dist'].mean():.2f}")
m5.metric("Anisotropy Ratio", f"{anisotropy_ratio:.2f}")

st.markdown("---")
left, right = st.columns([2, 1])
with left:
    st.subheader("📍 3D Spatial Distribution")
    fig3d = px.scatter_3d(
        df,
        x=x_col,
        y=y_col,
        z=z_col,
        color="region_label",
        hover_data=["cluster_id", "outlier_score", "pseudotime"],
        template="plotly_white",
    )
    fig3d.update_traces(marker=dict(size=4, opacity=0.8))
    st.plotly_chart(fig3d, use_container_width=True, theme=None, key="plot_3d_spatial")
with right:
    st.subheader("📈 Depth Profile")
    st.plotly_chart(px.histogram(df, x=z_col, nbins=25, template="plotly_white"), use_container_width=True, theme=None, key="plot_depth_profile")

c1, c2 = st.columns(2)
if vol_col:
    with c1:
        st.subheader("📊 Volume Distribution")
        st.plotly_chart(px.histogram(df, x=vol_col, nbins=30, template="plotly_white"), use_container_width=True, theme=None, key="plot_volume_distribution")
with c2:
    st.subheader("🧬 XY Projection Density")
    st.plotly_chart(px.density_heatmap(df, x=x_col, y=y_col, nbinsx=40, nbinsy=40, template="plotly_white"), use_container_width=True, theme=None, key="plot_xy_density")

st.markdown("---")
st.subheader("🧪 Developmental Stage Comparator & Cohort Dashboard")
cohort_rows = [
    {
        "sample": "uploaded_sample",
        "n": len(df),
        "mean_depth": float(df[z_col].mean()),
        "mean_nn_graph": float(df["nn_mean_dist"].mean()),
        "mean_volume": float(df[vol_col].mean()) if vol_col else np.nan,
    }
]
if comparison_files:
    for f in comparison_files:
        sdf = pd.read_csv(f)
        local_cols = {c.lower(): c for c in sdf.columns}
        sx, sy, sz = local_cols.get("centroid-2"), local_cols.get("centroid-1"), local_cols.get("centroid-0")
        if not (sx and sy and sz):
            continue
        sv = local_cols.get("volume_voxels")
        snn = local_cols.get("nearest_neighbor_dist")
        sdf = ensure_numeric(sdf, [sx, sy, sz] + ([sv] if sv else []) + ([snn] if snn else []))
        sdf = sdf.dropna(subset=[sx, sy, sz])
        if len(sdf) == 0:
            continue
        cohort_rows.append(
            {
                "sample": f.name,
                "n": len(sdf),
                "mean_depth": float(sdf[sz].mean()),
                "mean_nn_graph": float(sdf[snn].mean()) if snn else np.nan,
                "mean_volume": float(sdf[sv].mean()) if sv else np.nan,
            }
        )

cohort_df = pd.DataFrame(cohort_rows)
st.dataframe(cohort_df, use_container_width=True)
if len(cohort_df) > 1:
    metric_choice = st.selectbox("Comparator metric", ["mean_volume", "mean_depth", "mean_nn_graph"])
    comp = cohort_df[["sample", metric_choice]].dropna().copy()
    if len(comp) > 1:
        base = comp.iloc[0][metric_choice]
        comp["shift_vs_base"] = comp[metric_choice] - base
        comp["effect_size"] = comp["shift_vs_base"] / max(comp[metric_choice].std(ddof=0), 1e-9)
        st.plotly_chart(px.bar(comp, x="sample", y="shift_vs_base", color="effect_size", template="plotly_white"), use_container_width=True, theme=None, key="plot_cohort_shift")
        st.info(f"Most atypical sample: **{comp.iloc[comp['effect_size'].abs().argmax()]['sample']}**")

st.markdown("---")
st.subheader("🎯 Radial Morphometry from Landmark")
use_center = st.checkbox("Use auto landmark (global centroid)", value=True)
if use_center:
    lx, ly, lz = df[[x_col, y_col, z_col]].mean().tolist()
else:
    lx = st.number_input("Landmark X", float(df[x_col].min()), float(df[x_col].max()), float(df[x_col].mean()))
    ly = st.number_input("Landmark Y", float(df[y_col].min()), float(df[y_col].max()), float(df[y_col].mean()))
    lz = st.number_input("Landmark Z", float(df[z_col].min()), float(df[z_col].max()), float(df[z_col].mean()))

df["radial_distance"] = np.sqrt((df[x_col] - lx) ** 2 + (df[y_col] - ly) ** 2 + (df[z_col] - lz) ** 2)
st.plotly_chart(
    px.scatter(df, x="radial_distance", y="nn_mean_dist", color="region_label", template="plotly_white"),
    use_container_width=True,
    theme=None,
    key="plot_radial_morphometry",
)

st.markdown("---")
st.subheader("🖼️ Publication-Ready Figure Composer")
panels = st.multiselect("Select panels", ["Depth Hist", "Volume Hist", "Radial Scatter", "XY Projection"], default=["Depth Hist", "Radial Scatter"])
if panels:
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=panels)
    col_i = 1
    for panel in panels:
        if panel == "Depth Hist":
            h, b = np.histogram(df[z_col], bins=20)
            fig.add_trace(go.Bar(x=b[:-1], y=h, showlegend=False), row=1, col=col_i)
        elif panel == "Volume Hist" and vol_col:
            h, b = np.histogram(df[vol_col], bins=20)
            fig.add_trace(go.Bar(x=b[:-1], y=h, showlegend=False), row=1, col=col_i)
        elif panel == "Radial Scatter":
            fig.add_trace(go.Scatter(x=df["radial_distance"], y=df["nn_mean_dist"], mode="markers", showlegend=False), row=1, col=col_i)
        elif panel == "XY Projection":
            fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="markers", showlegend=False), row=1, col=col_i)
        col_i += 1
    fig.update_layout(template="plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True, theme=None, key="plot_publication_composer")

st.markdown("---")
st.subheader("🤖 Lightweight ML Phenotype Scoring")
if vol_col:
    volume_cut = st.slider("Reference threshold (volume percentile as altered phenotype)", 50, 99, 85)
    threshold = df[vol_col].quantile(volume_cut / 100)
    df["phenotype_label"] = np.where(df[vol_col] >= threshold, "altered", "normal")
    feature_scores = {}
    for feat in ["nn_mean_dist", "radial_distance", "pseudotime"]:
        means = df.groupby("phenotype_label")[feat].mean()
        if set(means.index) == {"normal", "altered"}:
            feature_scores[feat] = abs(float(means["altered"] - means["normal"]))
    if feature_scores:
        imp = pd.DataFrame({"feature": feature_scores.keys(), "importance": feature_scores.values()}).sort_values("importance", ascending=False)
        st.plotly_chart(px.bar(imp, x="feature", y="importance", template="plotly_white"), use_container_width=True, theme=None, key="plot_feature_importance")

st.subheader("⏱️ Temporal Trajectory (Pseudo-time)")
trend_df = df.copy()
trend_df["pt_bin"] = pd.cut(trend_df["pseudotime"], bins=10)
metrics = ["nn_mean_dist"] + ([vol_col] if vol_col else [])
trend = trend_df.groupby("pt_bin", observed=False)[metrics].mean().reset_index()
trend["pt_bin"] = trend["pt_bin"].astype(str)
st.plotly_chart(px.line(trend, x="pt_bin", y=metrics, template="plotly_white"), use_container_width=True, theme=None, key="plot_pseudotime_trend")

st.markdown("---")
st.subheader("🛡️ Quality-Control Audit Trail")
st.dataframe(qc_df, use_container_width=True)
st.download_button("Download QC report", qc_df.to_csv(index=False), "qc_report.csv", "text/csv")

st.subheader("🧭 Interactive Outlier Forensics")
top_n = st.slider("Show top outliers", 5, 50, 15)
outliers = df.nlargest(top_n, "outlier_score")
show_cols = [x_col, y_col, z_col, "outlier_score", "cluster_id", "region_label"] + ([vol_col] if vol_col else [])
st.dataframe(outliers[show_cols], use_container_width=True)
idx = st.selectbox("Inspect nucleus index", options=outliers.index.tolist())
st.json(df.loc[idx].fillna("NA").to_dict())

st.markdown("---")
st.subheader("📝 One-Click Methods Generator")
methods_text = make_methods_text(recipe)
st.markdown(methods_text)
st.download_button("Download methods text", methods_text, "methods.txt", "text/plain")

with st.expander("🔍 Inspect Processed Data"):
    st.dataframe(df, use_container_width=True)
