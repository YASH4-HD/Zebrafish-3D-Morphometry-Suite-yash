import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Zebrafish Analysis", layout="wide")

st.title("🔬 Zebrafish 3D Morphometry")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # --- DEBUGGING STEP ---
    # This will show us exactly what the columns are named
    st.write("Detected Columns:", list(df.columns))
    
    # Find all numeric columns automatically
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if len(numeric_cols) >= 3:
        # Use the first 3 numeric columns for X, Y, Z regardless of their names
        z_col = numeric_cols[0]
        y_col = numeric_cols[1]
        x_col = numeric_cols[2]
        vol_col = numeric_cols[3] if len(numeric_cols) > 3 else None

        # 1. Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Nuclei Count", len(df))
        m2.metric("Z-Axis", z_col)
        m3.metric("X-Axis", x_col)

        # 2. Charts with Forced Scaling
        c1, c2 = st.columns(2)
        
        with c1:
            fig_xy = px.scatter(df, x=x_col, y=y_col, color=z_col, 
                               title="Spatial Map (Top View)",
                               template="plotly_white")
            # This forces the graph to zoom into the actual data points
            fig_xy.update_xaxes(autorange=True)
            fig_xy.update_yaxes(autorange=True)
            st.plotly_chart(fig_xy, use_container_width=True)

        with c2:
            fig_z = px.histogram(df, x=z_col, title="Depth Distribution",
                                template="plotly_white", color_discrete_sequence=['#00CC96'])
            fig_z.update_xaxes(autorange=True)
            st.plotly_chart(fig_z, use_container_width=True)
            
        st.dataframe(df.head())
    else:
        st.error(f"Need at least 3 numeric columns. Found: {numeric_cols}")

else:
    st.info("Please upload your CSV file.")
