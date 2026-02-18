import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Zebrafish Morphometry", layout="wide")

st.title("🔬 Zebrafish 3D Morphometry Suite")
st.markdown("Quantitative analysis of nuclei spatial distribution.")

# 1. Sidebar
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Auto-detect columns
    cols = {c.lower(): c for c in df.columns}
    z_col = cols.get('centroid-0')
    y_col = cols.get('centroid-1')
    x_col = cols.get('centroid-2')
    vol_col = cols.get('volume_voxels')

    # 2. Key Metrics
    st.subheader("📊 Quantitative Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Nuclei", len(df))
    if vol_col: m2.metric("Avg Volume", f"{df[vol_col].mean():.1f}")
    if z_col: m3.metric("Z-Depth Span", f"{df[z_col].max() - df[z_col].min():.1f}")
    if cols.get('nearest_neighbor_dist'): 
        m4.metric("Mean NN Dist", f"{df[cols.get('nearest_neighbor_dist')].mean():.2f}")

    st.markdown("---")

    # 3. Scientific Projections (More reliable than 3D)
    st.subheader("📍 Spatial Projections")
    c1, c2 = st.columns(2)

    with c1:
        # Top-down View (XY)
        fig_xy = px.scatter(df, x=x_col, y=y_col, color=z_col, 
                           title="Top-down View (Color = Depth)",
                           color_continuous_scale='Viridis', template="plotly_white")
        st.plotly_chart(fig_xy, use_container_width=True)

    with c2:
        # Side View (XZ)
        fig_xz = px.scatter(df, x=x_col, y=z_col, color=vol_col,
                           title="Side View (Color = Volume)",
                           color_continuous_scale='Magma', template="plotly_white")
        st.plotly_chart(fig_xz, use_container_width=True)

    # 4. Depth Distribution
    st.subheader("📈 Depth Profile")
    fig_z = px.histogram(df, x=z_col, nbins=20, template="plotly_white", color_discrete_sequence=['#00CC96'])
    st.plotly_chart(fig_z, use_container_width=True)

    with st.expander("🔍 Inspect Raw Data"):
        st.dataframe(df)

else:
    st.info("👋 Please upload your CSV to begin.")
    # Static image for the landing page
    st.image("https://images.unsplash.com/photo-1576086213369-97a306d36557?auto=format&fit=crop&q=80&w=1000", caption="Computational Biology Analysis")
