import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import pandas as pd
import numpy as np
from PIL import Image
import gdown
import os

# Set Hugging Face token as an environment variable
os.environ["HUGGINGFACE_TOKEN"] = "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"

st.set_page_config(page_title="Text-to-X-ray Generator", layout="wide")

# Download dataset from Mega
try:
    gdown.download("https://mega.nz/file/J8hSxKKS#loXn1X-GcUhr5NSsgTTe5m7SrSw9Q9LbZV6KRX9U8W0", "preprocessed_data.pkl", quiet=False)
except Exception as e:
    st.error(f"Failed to download dataset: {e}")
    st.stop()

# Load model on CPU (use base Stable Diffusion model)
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float32,
        use_safetensors=True,
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
    )
    pipe.safety_checker = None
    st.write("Model loaded on CPU")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Load dataset
try:
    df = pd.read_pickle('preprocessed_data.pkl')
    st.sidebar.write(f"Loaded {len(df)} image-text pairs from dataset.")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

def numpy_to_pil(np_image):
    return Image.fromarray((np_image * 255).astype(np.uint8))

st.sidebar.header("Prompt Selection")
prompt_options = df['text'].tolist()
selected_prompt = st.sidebar.selectbox("Choose a dataset prompt", prompt_options)
custom_prompt = st.sidebar.text_input("Or enter custom prompt", selected_prompt)
prompt = custom_prompt if custom_prompt != selected_prompt else selected_prompt

st.title("Text-to-X-ray Generator")
st.markdown("Generate synthetic X-rays from text using a fine-tuned Stable Diffusion model.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Real X-ray")
    real_image_row = df[df['text'] == prompt]
    if not real_image_row.empty:
        real_image = numpy_to_pil(real_image_row.iloc[0]['image'])
        st.image(real_image, caption="From Dataset", use_container_width=True)
    else:
        st.write("No matching real image available.")
with col2:
    st.subheader("Generated X-ray")
    if st.button("Generate", key="generate"):
        with st.spinner("Generating..."):
            try:
                image = pipe(prompt, num_inference_steps=50).images[0]
                st.image(image, caption=f"Generated for: '{prompt}'", use_container_width=True)
            except Exception as e:
                st.error(f"Failed to generate image: {e}")
    else:
        st.write("Click 'Generate' to create an X-ray.")

st.markdown("---")
st.write("Built with Streamlit & Stable Diffusion | Dataset: IU X-ray | Created by Varun Kasa | © 2025")
