import plotly.io as pio
pio.renderers.default = "browser"
import json
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Zebrafish 3D Morphometry", layout="wide")

st.title("🔬 Zebrafish 3D Morphometry Suite")
st.markdown("Comprehensive analysis of 3D nuclei segmentation outputs.")


# ------------------------------
# Utility helpers
# ------------------------------
def compute_knn_metrics(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, k: int = 5):
    points = df[[x_col, y_col, z_col]].to_numpy(dtype=float)
    n = len(points)
    if n < 3:
        return pd.DataFrame(index=df.index)

    diff = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)

    k_eff = max(1, min(k, n - 1))
    nn_idx = np.argpartition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]
    nn_dists = np.take_along_axis(dists, nn_idx, axis=1)

    # Approximate local clustering: ratio of connected neighbor pairs under local radius
    clustering = []
    for i in range(n):
        neigh = nn_idx[i]
        if len(neigh) < 2:
            clustering.append(0.0)
            continue
        local = dists[np.ix_(neigh, neigh)]
        rad = float(np.median(nn_dists[i]) * 1.2)
        connected = np.sum((local < rad) & np.isfinite(local))
        possible = len(neigh) * (len(neigh) - 1)
        clustering.append(float(connected / possible) if possible else 0.0)

    metrics = pd.DataFrame(
        {
            "nn_mean_dist": nn_dists.mean(axis=1),
            "nn_degree": k_eff,
            "nn_clustering": clustering,
        },
        index=df.index,
    )
    return metrics


def simple_dbscan(points: np.ndarray, eps: float, min_pts: int):
    n = len(points)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    if n == 0:
        return labels

    diff = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diff, axis=2)

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
                j_neighbors = np.where(dists[j] <= eps)[0]
                if len(j_neighbors) >= min_pts:
                    seed.update(j_neighbors.tolist())
            if labels[j] == -1:
                labels[j] = cluster_id
            if labels[j] < 0:
                labels[j] = cluster_id

        cluster_id += 1

    return labels


def qc_report(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, vol_col: str | None):
    report = []
    duplicate_count = int(df.duplicated(subset=[x_col, y_col, z_col]).sum())
    report.append(("Duplicate centroids", duplicate_count, "warn" if duplicate_count else "pass"))

    coord_null = int(df[[x_col, y_col, z_col]].isna().sum().sum())
    report.append(("Missing centroid values", coord_null, "fail" if coord_null else "pass"))

    if vol_col:
        low_outliers = int((df[vol_col] <= 0).sum())
        high_outliers = int((df[vol_col] > df[vol_col].quantile(0.995)).sum())
        report.append(("Non-positive volume", low_outliers, "fail" if low_outliers else "pass"))
        report.append(("Extreme high volume", high_outliers, "warn" if high_outliers else "pass"))

    nn_col = "nearest_neighbor_dist" if "nearest_neighbor_dist" in df.columns else None
    if nn_col:
        bad_nn = int((df[nn_col] < 0).sum())
        report.append(("Negative NN distance", bad_nn, "fail" if bad_nn else "pass"))

    qc_df = pd.DataFrame(report, columns=["check", "count", "status"])
    return qc_df


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


def make_methods_text(settings: dict):
    lines = [
        "### Auto-generated Methods Summary",
        "Nuclei coordinates were analyzed in the Zebrafish 3D Morphometry Suite.",
        f"Input centroid columns: x={settings['x_col']}, y={settings['y_col']}, z={settings['z_col']}.",
        f"Outlier filtering mode: {settings['filter_mode']}.",
        f"k-nearest-neighbor setting: k={settings['k_neighbors']}.",
        f"Cluster detection parameters: eps={settings['eps']}, min_pts={settings['min_pts']}.",
        "Anatomical regions were assigned by axis-wise quantile/median partitioning.",
        "Pseudo-time was estimated via PCA first principal component on 3D coordinates.",
        "Quality-control checks included duplicate centroid detection, null coordinates, and volume outliers.",
    ]
    return "\n\n".join(lines)


# ==============================
# Sidebar – Data & Controls
# ==============================
st.sidebar.header("📂 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload nuclei CSV file", type="csv")
comparison_files = st.sidebar.file_uploader(
    "Optional: Upload comparison/stage CSVs", type="csv", accept_multiple_files=True
)

st.sidebar.header("⚙️ Analysis Settings")
filter_mode = st.sidebar.selectbox("Outlier filtering", ["None", "Volume top 1%", "Volume top 5%"])
k_neighbors = st.sidebar.slider("k for neighborhood graph", 2, 15, 5)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    cols = {c.lower(): c for c in df.columns}
    z_col = cols.get("centroid-0")
    y_col = cols.get("centroid-1")
    x_col = cols.get("centroid-2")
    vol_col = cols.get("volume_voxels")
    nn_col = cols.get("nearest_neighbor_dist")

    if not (x_col and y_col and z_col):
        st.error("⚠️ Required centroid columns not found in CSV.")
        st.stop()

    # Optional synthetic stress test mode
    with st.sidebar.expander("🧪 Synthetic data stress testing"):
        enable_synth = st.checkbox("Enable synthetic augmentation", value=False)
        synth_n = st.slider("Synthetic nuclei count", 50, 800, 120, 10)
        synth_noise = st.slider("Coordinate noise", 0.0, 60.0, 12.0, 1.0)

    if enable_synth:
        rng = np.random.default_rng(42)
        center = df[[x_col, y_col, z_col]].median().to_numpy()
        synthetic_xyz = rng.normal(loc=center, scale=synth_noise, size=(synth_n, 3))
        synthetic_df = pd.DataFrame(synthetic_xyz, columns=[x_col, y_col, z_col])
        synthetic_df["label"] = [f"synth_{i}" for i in range(synth_n)]
        if vol_col:
            synthetic_df[vol_col] = rng.lognormal(
                mean=np.log(max(df[vol_col].median(), 1.0)), sigma=0.35, size=synth_n
            )
        if nn_col:
            synthetic_df[nn_col] = np.nan
        synthetic_df["is_synthetic"] = True
        df["is_synthetic"] = False
        df = pd.concat([df, synthetic_df], ignore_index=True, sort=False)
        st.sidebar.success(f"Added {synth_n} synthetic nuclei for stress testing.")

    # Filtering
    df_filtered = df.copy()
    if vol_col and filter_mode != "None":
        q = 0.99 if filter_mode == "Volume top 1%" else 0.95
        df_filtered = df_filtered[df_filtered[vol_col] <= df_filtered[vol_col].quantile(q)].copy()

    # Feature 1: anatomical regions
    df_filtered = assign_anatomical_regions(df_filtered, x_col, y_col, z_col)

    # Feature 3: neighborhood graph metrics
    graph_df = compute_knn_metrics(df_filtered, x_col, y_col, z_col, k=k_neighbors)
    for c in graph_df.columns:
        df_filtered[c] = graph_df[c]

    # Feature 5: cluster detection
    eps_default = float(np.nanmedian(df_filtered["nn_mean_dist"]) * 1.5) if "nn_mean_dist" in df_filtered else 40.0
    st.sidebar.subheader("🧩 Cluster detection")
    eps = st.sidebar.slider("DBSCAN eps", 5.0, 200.0, float(np.clip(eps_default, 5.0, 200.0)), 1.0)
    min_pts = st.sidebar.slider("DBSCAN min points", 3, 20, 6)
    labels = simple_dbscan(df_filtered[[x_col, y_col, z_col]].to_numpy(dtype=float), eps=eps, min_pts=min_pts)
    df_filtered["cluster_id"] = labels

    # Feature 6: anisotropy
    coords = df_filtered[[x_col, y_col, z_col]].to_numpy(dtype=float)
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)[::-1]
    anisotropy_ratio = float(eigvals[0] / max(eigvals[-1], 1e-9))

    # Feature 14: pseudo-time using PCA component 1
    pc1 = centered @ eigvecs[:, -1]
    pc1_norm = (pc1 - pc1.min()) / max(pc1.max() - pc1.min(), 1e-9)
    df_filtered["pseudotime"] = pc1_norm

    # Feature 7: QC report
    qc_df = qc_report(df_filtered, x_col, y_col, z_col, vol_col)

    # Feature 8: outlier forensics
    if vol_col:
        vol_z = (df_filtered[vol_col] - df_filtered[vol_col].mean()) / max(df_filtered[vol_col].std(), 1e-9)
        nn_z = (df_filtered["nn_mean_dist"] - df_filtered["nn_mean_dist"].mean()) / max(
            df_filtered["nn_mean_dist"].std(), 1e-9
        )
        df_filtered["outlier_score"] = np.abs(vol_z) + np.abs(nn_z)
    else:
        nn_z = (df_filtered["nn_mean_dist"] - df_filtered["nn_mean_dist"].mean()) / max(
            df_filtered["nn_mean_dist"].std(), 1e-9
        )
        df_filtered["outlier_score"] = np.abs(nn_z)

    # Feature 11: analysis recipe
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
    recipe_file = st.sidebar.file_uploader("Load recipe JSON", type="json")
    if recipe_file:
        loaded = json.load(recipe_file)
        st.sidebar.info(f"Loaded recipe: {loaded}")

    # ==============================
    # Quantitative Summary
    # ==============================
    st.subheader("📊 Quantitative Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Nuclei", len(df_filtered))
    if vol_col:
        m2.metric("Avg Volume (voxels)", f"{df_filtered[vol_col].mean():.1f}")
    z_range = df_filtered[z_col].max() - df_filtered[z_col].min()
    m3.metric("Z-Depth Span", f"{z_range:.1f}")
    if nn_col:
        m4.metric("Mean NN Distance", f"{df_filtered[nn_col].mean():.2f}")
    m5.metric("Anisotropy Ratio", f"{anisotropy_ratio:.2f}")

    st.markdown("---")

    # ==============================
    # Core Visualizations
    # ==============================
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📍 3D Spatial Distribution")
        fig_3d = px.scatter_3d(
            df_filtered,
            x=x_col,
            y=y_col,
            z=z_col,
            color="region_label",
            symbol="cluster_id",
            hover_data=["outlier_score", "pseudotime"],
            template="ggplot2",
        )
        fig_3d.update_traces(marker=dict(size=4, opacity=0.8))
        st.plotly_chart(fig_3d, use_container_width=True, theme=None)

    with col_right:
        st.subheader("📈 Depth Profile")
        fig_z = px.histogram(df_filtered, x=z_col, nbins=25, color="region_label", template="plotly_white")
        st.plotly_chart(fig_z, use_container_width=True)

    c1, c2 = st.columns(2)
    if vol_col:
        with c1:
            st.subheader("📊 Volume Distribution")
            st.plotly_chart(px.histogram(df_filtered, x=vol_col, nbins=30, template="plotly_white"), use_container_width=True)
    with c2:
        st.subheader("🧬 XY Projection Density")
        st.plotly_chart(
            px.density_heatmap(df_filtered, x=x_col, y=y_col, nbinsx=40, nbinsy=40, template="plotly_white"),
            use_container_width=True,
        )

    # Feature 2 + 10: comparison and batch cohort dashboard
    st.markdown("---")
    st.subheader("🧪 Developmental Stage Comparator & Cohort Dashboard")
    cohort_rows = []
    base_stats = {
        "sample": "uploaded_sample",
        "n": len(df_filtered),
        "mean_depth": float(df_filtered[z_col].mean()),
        "mean_nn_graph": float(df_filtered["nn_mean_dist"].mean()),
        "mean_volume": float(df_filtered[vol_col].mean()) if vol_col else np.nan,
    }
    cohort_rows.append(base_stats)

    if comparison_files:
        for f in comparison_files:
            sdf = pd.read_csv(f)
            if all(c in sdf.columns for c in [x_col, y_col, z_col]):
                row = {
                    "sample": f.name,
                    "n": len(sdf),
                    "mean_depth": float(sdf[z_col].mean()),
                    "mean_nn_graph": float(sdf[nn_col].mean()) if nn_col and nn_col in sdf.columns else np.nan,
                    "mean_volume": float(sdf[vol_col].mean()) if vol_col and vol_col in sdf.columns else np.nan,
                }
                cohort_rows.append(row)

    cohort_df = pd.DataFrame(cohort_rows)
    st.dataframe(cohort_df, use_container_width=True)

    if len(cohort_df) > 1:
        metric_choice = st.selectbox("Comparator metric", ["mean_volume", "mean_depth", "mean_nn_graph"])
        comp = cohort_df[["sample", metric_choice]].dropna()
        if len(comp) > 1:
            base_val = comp.iloc[0][metric_choice]
            comp["shift_vs_base"] = comp[metric_choice] - base_val
            comp["effect_size"] = comp["shift_vs_base"] / max(comp[metric_choice].std(), 1e-9)
            st.plotly_chart(px.bar(comp, x="sample", y="shift_vs_base", color="effect_size", template="plotly_white"), use_container_width=True)
            st.info(f"Most atypical sample: **{comp.iloc[comp['effect_size'].abs().argmax()]['sample']}**")

    # Feature 4: radial morphometry
    st.markdown("---")
    st.subheader("🎯 Radial Morphometry from Landmark")
    use_center = st.checkbox("Use auto landmark (global centroid)", value=True)
    if use_center:
        lx, ly, lz = df_filtered[[x_col, y_col, z_col]].mean().tolist()
    else:
        lx = st.number_input("Landmark X", float(df_filtered[x_col].min()), float(df_filtered[x_col].max()), float(df_filtered[x_col].mean()))
        ly = st.number_input("Landmark Y", float(df_filtered[y_col].min()), float(df_filtered[y_col].max()), float(df_filtered[y_col].mean()))
        lz = st.number_input("Landmark Z", float(df_filtered[z_col].min()), float(df_filtered[z_col].max()), float(df_filtered[z_col].mean()))
    radial = np.sqrt((df_filtered[x_col] - lx) ** 2 + (df_filtered[y_col] - ly) ** 2 + (df_filtered[z_col] - lz) ** 2)
    df_filtered["radial_distance"] = radial
    radial_fig = px.scatter(df_filtered, x="radial_distance", y="nn_mean_dist", color="region_label", template="plotly_white")
    st.plotly_chart(radial_fig, use_container_width=True)

    # Feature 12: publication figure composer
    st.markdown("---")
    st.subheader("🖼️ Publication-Ready Figure Composer")
    panels = st.multiselect(
        "Select panels", ["3D", "Depth Hist", "Volume Hist", "Radial Scatter"], default=["Depth Hist", "Radial Scatter"]
    )
    if panels:
        fig = make_subplots(rows=1, cols=len(panels), subplot_titles=panels)
        col_i = 1
        for panel in panels:
            if panel == "Depth Hist":
                hist, bins = np.histogram(df_filtered[z_col], bins=20)
                fig.add_trace(go.Bar(x=bins[:-1], y=hist, name=panel, showlegend=False), row=1, col=col_i)
            elif panel == "Volume Hist" and vol_col:
                hist, bins = np.histogram(df_filtered[vol_col], bins=20)
                fig.add_trace(go.Bar(x=bins[:-1], y=hist, name=panel, showlegend=False), row=1, col=col_i)
            elif panel == "Radial Scatter":
                fig.add_trace(
                    go.Scatter(x=df_filtered["radial_distance"], y=df_filtered["nn_mean_dist"], mode="markers", name=panel, showlegend=False),
                    row=1,
                    col=col_i,
                )
            elif panel == "3D":
                # 3D shown in dedicated panel above; add 2D projection for composed figure
                fig.add_trace(
                    go.Scatter(x=df_filtered[x_col], y=df_filtered[y_col], mode="markers", name="XY projection", showlegend=False),
                    row=1,
                    col=col_i,
                )
            col_i += 1
        fig.update_layout(template="plotly_white", height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Feature 13: lightweight phenotype scoring
    st.markdown("---")
    st.subheader("🤖 Lightweight ML Phenotype Scoring")
    if vol_col:
        volume_cut = st.slider(
            "Reference threshold (volume percentile as altered phenotype)", 50, 99, 85
        )
        threshold = df_filtered[vol_col].quantile(volume_cut / 100)
        df_filtered["phenotype_label"] = np.where(df_filtered[vol_col] >= threshold, "altered", "normal")
        feats = ["nn_mean_dist", "radial_distance", "pseudotime"]
        feat_imp = {}
        for feat in feats:
            g = df_filtered.groupby("phenotype_label")[feat].mean()
            if set(g.index) == {"normal", "altered"}:
                feat_imp[feat] = abs(float(g["altered"] - g["normal"]))
        imp_df = pd.DataFrame({"feature": list(feat_imp), "importance": list(feat_imp.values())}).sort_values("importance", ascending=False)
        st.plotly_chart(px.bar(imp_df, x="feature", y="importance", template="plotly_white"), use_container_width=True)

    # Feature 14 trend view
    st.subheader("⏱️ Temporal Trajectory (Pseudo-time)")
    bin_df = df_filtered.copy()
    bin_df["pt_bin"] = pd.cut(bin_df["pseudotime"], bins=10)
    trend = bin_df.groupby("pt_bin", observed=False)[["nn_mean_dist"] + ([vol_col] if vol_col else [])].mean().reset_index()
    trend["pt_bin"] = trend["pt_bin"].astype(str)
    st.plotly_chart(px.line(trend, x="pt_bin", y=[c for c in trend.columns if c != "pt_bin"], template="plotly_white"), use_container_width=True)

    # Feature 7 + 8 QC and outlier panel
    st.markdown("---")
    st.subheader("🛡️ Quality-Control Audit Trail")
    st.dataframe(qc_df, use_container_width=True)
    qc_csv = qc_df.to_csv(index=False)
    st.download_button("Download QC report", qc_csv, "qc_report.csv", "text/csv")

    st.subheader("🧭 Interactive Outlier Forensics")
    top_n = st.slider("Show top outliers", 5, 50, 15)
    outliers = df_filtered.nlargest(top_n, "outlier_score")
    st.dataframe(outliers[[x_col, y_col, z_col, "outlier_score", "cluster_id", "region_label"] + ([vol_col] if vol_col else [])])
    selected_idx = st.selectbox("Inspect nucleus index", options=outliers.index.tolist())
    st.json(df_filtered.loc[selected_idx].fillna("NA").to_dict())

    # Feature 15 methods text
    st.markdown("---")
    st.subheader("📝 One-Click Methods Generator")
    methods_text = make_methods_text(recipe)
    st.markdown(methods_text)
    st.download_button("Download methods text", methods_text, "methods.txt", "text/plain")

    # Raw data
    with st.expander("🔍 Inspect Processed Data"):
        st.dataframe(df_filtered, use_container_width=True)

else:
    st.info("👋 Please upload your segmentation CSV file to begin.")
