
import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import pandas as pd
import numpy as np
from PIL import Image

st.set_page_config(page_title="Text-to-X-ray Generator", layout="wide")

# Load model on CPU
pipe = StableDiffusionPipeline.from_pretrained(
    "fine_tuned_xray_model_full_fixed_v2",
    torch_dtype=torch.float32,
    use_safetensors=True
)
pipe.safety_checker = None
st.write("Model loaded on CPU")

# Load dataset
df = pd.read_pickle('preprocessed_data.pkl')
st.sidebar.write(f"Loaded {len(df)} image-text pairs from dataset.")

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
            image = pipe(prompt, num_inference_steps=10).images[0]
        st.image(image, caption=f"Generated for: '{prompt}'", use_container_width=True)
    else:
        st.write("Click 'Generate' to create an X-ray.")

st.markdown("---")
st.write("Built with Streamlit & Stable Diffusion | Dataset: IU X-ray | Created by Varun Kasa | © 2025")
