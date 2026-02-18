import streamlit as st
import pandas as pd

st.title("🔬 Zebrafish Research Dashboard")
st.write("If you see this, the app is working!")

uploaded_file = st.file_uploader("Upload your CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
