import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
import torch
import os
import requests

# Set Streamlit page configuration
st.set_page_config(page_title="Text-to-X-ray Generator", layout="wide")

# Function to download the dataset if not present
def download_dataset():
    dataset_path = "preprocessed_data.pkl"
    dataset_url = "https://drive.google.com/uc?export=download&id=1VgYUzoYNS5FMC4mjIVA8EyIiUZf5pJFN"  # Updated with your Google Drive URL

    if not os.path.exists(dataset_path):
        st.write("Downloading dataset...")
        try:
            response = requests.get(dataset_url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            with open(dataset_path, "wb") as f:
                f.write(response.content)
            st.write("Dataset downloaded successfully.")
        except Exception as e:
            st.error(f"Failed to download dataset: {e}")
            st.stop()

# Load dataset from local path or download it
try:
    download_dataset()
    df = pd.read_pickle("preprocessed_data.pkl")
    st.sidebar.write(f"Loaded {len(df)} image-text pairs from dataset.")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Load pre-trained model on CPU
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",  # Updated model
        torch_dtype=torch.float32,  # Keep CPU compatibility
        use_safetensors=True
    )
    pipe.safety_checker = None
    st.write("Pre-trained model (stabilityai/stable-diffusion-2-1) loaded on CPU.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    pipe = None

def numpy_to_pil(np_image):
    return Image.fromarray((np_image * 255).astype(np.uint8))

# Sidebar for prompt selection
st.sidebar.header("Prompt Selection")
prompt_options = df['text'].tolist()
selected_prompt = st.sidebar.selectbox("Choose a dataset prompt", prompt_options)
custom_prompt = st.sidebar.text_input("Or enter custom prompt", selected_prompt)
prompt = custom_prompt if custom_prompt != custom_prompt else selected_prompt

# Enhance the prompt with additional details for better chest X-ray generation
if prompt:
    prompt = f"{prompt} in grayscale, anatomical structures visible, clear lung fields, realistic X-ray style, black and white, high contrast, medical imaging, chest X-ray with visible heart, lungs, ribs, and diaphragm"

# Main app layout
st.title("Text-to-X-ray Generator")
st.markdown("Generate synthetic X-rays from text using a pre-trained Stable Diffusion 2.1 model.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Real X-ray")
    real_image_row = df[df['text'] == selected_prompt]
    if not real_image_row.empty:
        real_image = numpy_to_pil(real_image_row.iloc[0]['image'])
        st.image(real_image, caption="From Dataset", use_container_width=True)
    else:
        st.write("No matching real image available.")
with col2:
    st.subheader("Generated X-ray")
    if pipe is None:
        st.write("Image generation is disabled due to model loading failure.")
    else:
        st.write("Note: Using a pre-trained model (stabilityai/stable-diffusion-2-1); generated images may not be perfectly accurate X-rays.")
        if st.button("Generate", key="generate"):
            with st.spinner("Generating... (This may take a while on CPU)"):
                try:
                    # Generate image with specified resolution to match real image (512x512)
                    image = pipe(
                        prompt,
                        num_inference_steps=10,
                        height=512,
                        width=512
                    ).images[0]
                    st.image(image, caption=f"Generated for: '{prompt}'", use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to generate image: {e}")
        else:
            st.write("Click 'Generate' to create an image.")

# Footer
st.markdown("---")
st.write("Built with Streamlit & Stable Diffusion 2.1 | Dataset: IU X-ray | Created by Varun Kasa | © 2025")
