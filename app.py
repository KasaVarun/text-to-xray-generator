import streamlit as st

st.set_page_config(page_title="Text-to-X-ray Generator", layout="wide")

st.title("Text-to-X-ray Generator")
st.markdown("Generate synthetic X-rays from text using a fine-tuned Stable Diffusion model.")

st.sidebar.header("Prompt Selection")
custom_prompt = st.sidebar.text_input("Enter a custom prompt", "Frontal chest X-ray showing cardiomegaly")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Real X-ray")
    st.write("Dataset loading is temporarily disabled due to deployment issues.")
with col2:
    st.subheader("Generated X-ray")
    st.write("Image generation is temporarily disabled due to deployment issues.")

st.markdown("---")
st.write("Built with Streamlit & Stable Diffusion | Dataset: IU X-ray | Created by Varun Kasa | © 2025")
