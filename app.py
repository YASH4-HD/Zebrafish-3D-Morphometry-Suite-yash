import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Zebrafish 3D Morphometry", layout="wide")

st.title("🔬 Zebrafish 3D Morphometry Suite")
st.markdown("Developed for quantitative analysis of nuclei distribution.")

# 1. Sidebar for Data Upload
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # --- AUTO-DETECT COLUMNS ---
    # This finds columns regardless of capitalization (e.g., 'Centroid-0' or 'centroid-0')
    cols = {c.lower(): c for c in df.columns}
    z_col = cols.get('centroid-0')
    y_col = cols.get('centroid-1')
    x_col = cols.get('centroid-2')
    vol_col = cols.get('volume_voxels')
    nn_col = cols.get('nearest_neighbor_dist')

    # 2. Key Metrics
    st.subheader("📊 Quantitative Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Nuclei", len(df))
    
    if vol_col:
        m2.metric("Avg Volume", f"{df[vol_col].mean():.1f}")
    if z_col:
        z_range = df[z_col].max() - df[z_col].min()
        m3.metric("Z-Depth Span", f"{z_range:.1f}")
    if nn_col:
        m4.metric("Mean NN Dist", f"{df[nn_col].mean():.2f}")

    st.markdown("---")

    # 3. Visualizations
    col_left, col_right = st.columns([2, 1])

    if x_col and y_col and z_col:
        with col_left:
            st.subheader("📍 3D Spatial Distribution")
            fig_3d = px.scatter_3d(
                df, x=x_col, y=y_col, z=z_col,
                color=vol_col if vol_col else None,
                template="plotly_white",
                title="3D Nuclei Map"
            )
            fig_3d.update_traces(marker=dict(size=3, opacity=0.8))
            st.plotly_chart(fig_3d, use_container_width=True)

        with col_right:
            st.subheader("📈 Depth Profile")
            fig_z = px.histogram(
                df, x=z_col, 
                nbins=20, 
                color_discrete_sequence=['#00CC96'],
                template="plotly_white"
            )
            st.plotly_chart(fig_z, use_container_width=True)
    else:
        st.error("⚠️ Error: Could not find Centroid columns in CSV. Please check column names.")

    with st.expander("🔍 Inspect Raw Data"):
        st.dataframe(df)

else:
    st.info("👋 Please upload your CSV to begin.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Zebrafish_embryo_72h.jpg/800px-Zebrafish_embryo_72h.jpg")
