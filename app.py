import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import torch
import os
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(page_title="Text-to-X-ray Generator", layout="wide")

# Function to download the dataset if not present
def download_dataset():
    dataset_path = "preprocessed_data_subset.pkl"
    dataset_url = "https://drive.google.com/file/d/1VgYUzoYNS5FMC4mjIVA8EyIiUZf5pJFN/view?usp=sharing"  # Replace with your Google Drive direct download URL for preprocessed_data_subset.pkl

    if not os.path.exists(dataset_path):
        st.info("Downloading dataset (subset)...")
        logger.info(f"Downloading dataset from {dataset_url}")
        try:
            response = requests.get(dataset_url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            with open(dataset_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            st.success("Dataset downloaded successfully.")
            logger.info("Dataset downloaded successfully.")
        except Exception as e:
            st.error(f"Failed to download dataset: {str(e)}")
            logger.error(f"Failed to download dataset: {str(e)}")
            st.stop()

# Load dataset from local path or download it
try:
    download_dataset()
    df = pd.read_pickle("preprocessed_data_subset.pkl")
    st.sidebar.write(f"Loaded {len(df)} image-text pairs from dataset.")
    logger.info(f"Loaded {len(df)} image-text pairs from dataset.")
except Exception as e:
    st.error(f"Failed to load dataset: {str(e)}")
    logger.error(f"Failed to load dataset: {str(e)}")
    st.stop()

# Load pre-trained model on CPU
try:
    logger.info("Loading Stable Diffusion model...")
    from diffusers import StableDiffusionPipeline  # Import here to ensure diffusers is installed
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",  # Updated model
        torch_dtype=torch.float32,  # Keep CPU compatibility
        use_safetensors=True
    )
    pipe.safety_checker = None
    st.write("Pre-trained model (stabilityai/stable-diffusion-2-1) loaded on CPU.")
    logger.info("Model loaded successfully.")
except ImportError as e:
    st.error("Failed to import StableDiffusionPipeline. Ensure 'diffusers' is installed correctly.")
    logger.error(f"ImportError: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    logger.error(f"Failed to load model: {str(e)}")
    st.stop()

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
                        num_inference_steps=5,  # Reduced for faster generation
                        height=512,
                        width=512
                    ).images[0]
                    st.image(image, caption=f"Generated for: '{prompt}'", use_container_width=True)
                    logger.info(f"Generated image for prompt: {prompt}")
                except Exception as e:
                    st.error(f"Failed to generate image: {str(e)}")
                    logger.error(f"Failed to generate image: {str(e)}")
        else:
            st.write("Click 'Generate' to create an image.")

# Footer
st.markdown("---")
st.write("Built with Streamlit & Stable Diffusion 2.1 | Dataset: IU X-ray | Created by Varun Kasa | © 2025")
