import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Zebrafish 3D Morphometry", layout="wide")

st.title("🔬 Zebrafish 3D Morphometry Suite")
st.markdown("Quantitative analysis of 3D nuclei segmentation outputs.")

# ==============================
# Sidebar – Data Upload
# ==============================

st.sidebar.header("📂 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload nuclei CSV file", type="csv")

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # ------------------------------
    # Auto-detect important columns
    # ------------------------------
    cols = {c.lower(): c for c in df.columns}

    z_col = cols.get('centroid-0')
    y_col = cols.get('centroid-1')
    x_col = cols.get('centroid-2')
    vol_col = cols.get('volume_voxels')
    nn_col = cols.get('nearest_neighbor_dist')

    if not (x_col and y_col and z_col):
        st.error("⚠️ Required centroid columns not found in CSV.")
        st.stop()

    # ------------------------------
    # Optional Outlier Filtering
    # ------------------------------
    if vol_col:
        df_filtered = df[df[vol_col] < df[vol_col].quantile(0.99)]
    else:
        df_filtered = df

    # ==============================
    # Quantitative Summary
    # ==============================

    st.subheader("📊 Quantitative Summary")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Total Nuclei", len(df_filtered))

    if vol_col:
        m2.metric("Avg Volume (voxels)",
                  f"{df_filtered[vol_col].mean():.1f}")

    z_range = df_filtered[z_col].max() - df_filtered[z_col].min()
    m3.metric("Z-Depth Span", f"{z_range:.1f}")

    if nn_col:
        m4.metric("Mean NN Distance",
                  f"{df_filtered[nn_col].mean():.2f}")

    st.markdown("---")

    # ==============================
    # Visualizations
    # ==============================

    col_left, col_right = st.columns([2, 1])

    # ---- 3D Spatial Distribution ----
    with col_left:
        st.subheader("📍 3D Spatial Distribution")

        fig_3d = px.scatter_3d(
            df_filtered,
            x=x_col,
            y=y_col,
            z=z_col,
            color=vol_col if vol_col else None,
            template="plotly_white"
        )

        fig_3d.update_traces(marker=dict(size=3, opacity=0.8))
        st.plotly_chart(fig_3d, use_container_width=True)

    # ---- Depth Profile ----
    with col_right:
        st.subheader("📈 Depth Profile")

        fig_z = px.histogram(
            df_filtered,
            x=z_col,
            nbins=25,
            color_discrete_sequence=['#00CC96'],
            template="plotly_white"
        )

        st.plotly_chart(fig_z, use_container_width=True)

    # ==============================
    # Additional Morphometry Plots
    # ==============================

    st.markdown("---")

    col1, col2 = st.columns(2)

    # ---- Volume Distribution ----
    if vol_col:
        with col1:
            st.subheader("📊 Volume Distribution")

            fig_vol = px.histogram(
                df_filtered,
                x=vol_col,
                nbins=30,
                template="plotly_white"
            )

            st.plotly_chart(fig_vol, use_container_width=True)

    # ---- XY Density Map ----
    with col2:
        st.subheader("🧬 XY Projection Density")

        fig_xy = px.density_heatmap(
            df_filtered,
            x=x_col,
            y=y_col,
            nbinsx=40,
            nbinsy=40,
            template="plotly_white"
        )

        st.plotly_chart(fig_xy, use_container_width=True)

    # ==============================
    # Raw Data Viewer
    # ==============================

    with st.expander("🔍 Inspect Raw Data"):
        st.dataframe(df_filtered)

else:
    st.info("👋 Please upload your segmentation CSV file to begin.")
