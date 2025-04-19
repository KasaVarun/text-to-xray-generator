import os
import tarfile
import requests
import pandas as pd
import cv2
import numpy as np
from xml.etree import ElementTree as ET
from transformers import CLIPTokenizer

# Define URLs
image_url = 'https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz'
report_url = 'https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz'

# Download datasets
print("Downloading datasets...")
os.system(f"wget {image_url} -O NLMCXR_png.tgz")
os.system(f"wget {report_url} -O NLMCXR_reports.tgz")

# Extract images
print("Extracting images...")
with tarfile.open('NLMCXR_png.tgz', 'r:gz') as tar:
    tar.extractall('NLMCXR_png')

# Extract reports
print("Extracting reports...")
with tarfile.open('NLMCXR_reports.tgz', 'r:gz') as tar:
    tar.extractall('NLMCXR_reports')

# Define directories
report_dir = 'NLMCXR_reports/ecgen-radiology'
image_dir = 'NLMCXR_png'

# Verify directories
xml_count = len([f for f in os.listdir(report_dir) if f.endswith('.xml')])
png_count = len([f for f in os.listdir(image_dir) if f.endswith('.png')])
print(f"Reports in {report_dir}: {xml_count} files")
print(f"Images in {image_dir}: {png_count} files")
print(f"Sample PNG files: {os.listdir(image_dir)[:5]}")

# Parse XML reports
def parse_report(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        parent_images = root.findall('.//parentImage')

        image_ids = [img.get('id') for img in parent_images if img.get('id')]

        findings = root.find('.//AbstractText[@Label="FINDINGS"]')
        impression = root.find('.//AbstractText[@Label="IMPRESSION"]')
        text = (impression.text if impression is not None and impression.text else
                findings.text if findings is not None and findings.text else '')
        return image_ids, text
    except ET.ParseError as e:
        print(f"Error parsing {xml_file}: {e}")
        return [], ""

# Collect image-text pairs
data = []
xml_files = [f for f in os.listdir(report_dir) if f.endswith('.xml')]
print(f"Processing {len(xml_files)} XML files...")

for report_file in xml_files:
    xml_path = os.path.join(report_dir, report_file)
    image_ids, text = parse_report(xml_path)
    for image_id in image_ids:
        image_path = os.path.join(image_dir, f'{image_id}.png')
        if os.path.exists(image_path):
            data.append({'image_path': image_path, 'text': text})

print(f"\nCollected {len(data)} image-text pairs.")

# Create DataFrame
df = pd.DataFrame(data)
print("Dataset sample:")
print(df.head())

# Preprocessing functions
def preprocess_image(image_path, size=(256, 256)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.resize(image, size)
    return image / 255.0

# Truncate prompts to 75 tokens for CLIP
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
def truncate_prompt(text, max_length=75):
    try:
        encoded = tokenizer.encode(text, max_length=max_length, truncation=True, padding=False, add_special_tokens=True)
        return tokenizer.decode(encoded, skip_special_tokens=True)
    except Exception as e:
        print(f"Error truncating prompt: {e}")
        return text[:50]

# Apply preprocessing
if not df.empty:
    df['image'] = df['image_path'].apply(preprocess_image)
    df['text'] = df['text'].apply(truncate_prompt)
    df.to_pickle('preprocessed_data.pkl')
    print("Data preparation complete. Saved as 'preprocessed_data.pkl'.")
else:
    print("DataFrame is empty. Check XML parsing and PNG file matching.")