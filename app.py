import streamlit as st
import os
import requests
import logging
import numpy as np
import pandas as pd
from PIL import Image
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page configuration
st.set_page_config(page_title="Text-to-X-ray Generator", layout="wide")

# Function to download the dataset if not present
def download_dataset():
    dataset_path = "preprocessed_data_subset.pkl"
    dataset_url = "https://drive.google.com/uc?export=download&id=1XM-6yIl2yc9qT5H8rS2NPa-hkB-ybodC"  # Updated direct download URL for preprocessed_data_subset.pkl

    if not os.path.exists(dataset_path):
        st.info("Downloading dataset (subset)...")
        logger.info(f"Downloading dataset from {dataset_url}")
        try:
            # First request to get the confirmation token
            session = requests.Session()
            response = session.get(dataset_url, stream=True)
            response.raise_for_status()

            # Check if the response is an HTML page (indicating a confirmation is needed)
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                # Extract the confirmation token from the HTML
                html_content = response.text
                token_start = html_content.find('confirm=') + 8
                token_end = html_content.find('&', token_start)
                if token_end == -1:
                    token_end = len(html_content)
                confirm_token = html_content[token_start:token_end]

                # Make a second request with the confirmation token
                download_url = f"{dataset_url}&confirm={confirm_token}"
                response = session.get(download_url, stream=True)
                response.raise_for_status()

            # Download the file in chunks
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
    logger.info(f"Numpy version: {np.__version__}")
    logger.info(f"Pandas version: {pd.__version__}")
    df = pd.read_pickle("preprocessed_data_subset.pkl")
    st.sidebar.write(f"Loaded {len(df)} image-text pairs from dataset.")
    logger.info(f"Loaded {len(df)} image-text pairs from dataset.")
except ImportError as e:
    st.error(f"Failed to import dependency: {str(e)}")
    logger.error(f"ImportError: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"Failed to load dataset: {str(e)}")
    logger.error(f"Failed to load dataset: {str(e)}")
    st.stop()

try:
    logger.info("Loading Stable Diffusion model...")
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float32,
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
        st.image(real_image, caption="From Dataset", use_column_width=True)
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
                    st.image(image, caption=f"Generated for: '{prompt}'", use_column_width=True)
                    logger.info(f"Generated image for prompt: {prompt}")
                except Exception as e:
                    st.error(f"Failed to generate image: {str(e)}")
                    logger.error(f"Failed to generate image: {str(e)}")
        else:
            st.write("Click 'Generate' to create an image.")

# Footer
st.markdown("---")
st.write("Built with Streamlit & Stable Diffusion 2.1 | Dataset: IU X-ray | Created by Varun Kasa | © 2025")
