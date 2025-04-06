
# Text-to-X-ray Generator

## Overview
Fine-tunes Stable Diffusion on IU X-ray dataset (7470 pairs) with 4 epochs. Note: Due to a placeholder loss, fine-tuning was limited, reflected in the SSIM.

## Dataset
- Images: https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz
- Reports: https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz

## Usage
1. Visit the hosted app: [Streamlit URL] (to be updated after hosting).
2. Or clone the repository from [GitHub URL] (to be hosted).
3. Install dependencies: `pip install -r requirements.txt`.
4. Run locally: `streamlit run app.py`.

## Results
- Average SSIM (7470 samples): 0.1989
- Compute Units: ~20-24 for training, ~20-30 for full evaluation on A100

## Author
- Created by Varun Kasa
