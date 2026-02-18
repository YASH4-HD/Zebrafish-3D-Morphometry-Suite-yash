import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Zebrafish 3D Analysis", layout="wide")

st.title("🔬 Zebrafish 3D Morphometry Suite")
st.markdown("Quantitative analysis of nuclei spatial distribution in zebrafish embryos.")

# 1. Sidebar for Data Upload
st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload 'zebrafish_nuclei_data.csv'", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # EXACT MAPPING based on your detected columns
    z_col = 'centroid-0'
    y_col = 'centroid-1'
    x_col = 'centroid-2'
    vol_col = 'volume_voxels'
    nn_col = 'nearest_neighbor_dist'

    # 2. Key Metrics Row
    st.subheader("📊 Quantitative Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Nuclei", len(df))
    m2.metric("Avg Volume", f"{df[vol_col].mean():.1f}")
    m3.metric("Z-Depth Span", f"{df[z_col].max() - df[z_col].min():.1f}")
    m4.metric("Mean NN Dist", f"{df[nn_col].mean():.2f}")

    st.markdown("---")

    # 3. 3D Visualization
    st.subheader("📍 3D Spatial Distribution")
    fig_3d = px.scatter_3d(
        df, x=x_col, y=y_col, z=z_col,
        color=vol_col,
        opacity=0.8,
        color_continuous_scale='Viridis',
        labels={x_col: 'X (microns)', y_col: 'Y (microns)', z_col: 'Z (Depth)'}
    )
    # This ensures the camera is positioned to see the data immediately
    fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), scene=dict(aspectmode='data'))
    st.plotly_chart(fig_3d, use_container_width=True)

    # 4. Depth Profile Histogram
    st.subheader("📈 Nuclei Density by Depth")
    fig_z = px.histogram(
        df, x=z_col, 
        nbins=20, 
        color_discrete_sequence=['#00CC96'],
        labels={z_col: 'Z-Coordinate (Depth)'},
        template="plotly_white"
    )
    st.plotly_chart(fig_z, use_container_width=True)

    # 5. Raw Data Inspection
    with st.expander("🔍 Inspect Raw Morphometric Data"):
        st.dataframe(df)

else:
    st.info("👋 Welcome! Please upload your CSV to visualize the 3D data.")
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/Zebrafish_embryo_72h.jpg/800px-Zebrafish_embryo_72h.jpg", caption="Zebrafish Embryo Model")
