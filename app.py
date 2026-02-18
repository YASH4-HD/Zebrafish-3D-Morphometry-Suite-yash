import streamlit as st
import pandas as pd
import plotly.express as px

# Force a simple layout to bypass Altair internal errors
st.set_page_config(page_title="Zebrafish Suite", layout="centered")

st.title("🔬 Zebrafish 3D Analysis")
st.success("The app is officially online!")

st.write("Upload your data below:")
uploaded_file = st.file_uploader("Choose CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(df.head())
