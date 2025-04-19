## Instructions to Run the Code

To run the Text-to-X-ray Generator, follow these detailed steps for installation, dataset setup, and launching the application. These instructions assume you have basic familiarity with Python and command-line tools.

### Prerequisites

Before starting, ensure you have the following installed:
- **Python**: Version 3.8 or higher.
- **Git**: For cloning the repository.
- **Virtual Environment** (optional but recommended): To isolate project dependencies.

### Step 1: Installation

1. **Clone the Repository**  
   Open a terminal and run the following command to download the project files:
   ```bash
   git clone https://github.com/KasaVarun/text-to-xray-generator.git
   cd text-to-xray-generator
   ```

2. **Set Up a Virtual Environment** (Recommended)  
   This step keeps the project’s dependencies separate from your system’s Python environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**  
   Install the required Python libraries using the provided `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```
   Key libraries installed include:
   - `streamlit`: For the interactive web application.
   - `pandas`, `numpy`: For data handling.
   - `pillow`: For image processing.
   - `torch`, `diffusers`, `transformers`: For running the Stable Diffusion model.

### Step 2: Dataset Setup

The project uses the **IU X-ray dataset (NLMCXR)**, which includes chest X-ray images and their corresponding text reports. Follow these steps to prepare the dataset:

1. **Download the Dataset**  
   - **Images**: Download from [NLMCXR_png.tgz](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz).  
   - **Reports**: Download from [NLMCXR_reports.tgz](https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz).

2. **Extract the Files**  
   - Unzip the images into a folder named `NLMCXR_png` in the project directory.  
   - Unzip the reports into a folder named `NLMCXR_reports` in the project directory.

3. **Preprocess the Dataset**  
   Run the preprocessing script to prepare the data for the model:
   ```bash
   python preprocess_data.py
   ```
   This script will:
   - Parse the XML reports to extract image-text pairs.
   - Process the images (resize, convert to grayscale, normalize).
   - Truncate text prompts to 75 tokens.
   - Save the processed data as `preprocessed_data.pkl` in the project directory.

### Step 3: Running the Application

Once the setup is complete, you can launch the Streamlit application to generate synthetic X-rays.

1. **Ensure the Fine-Tuned Model is Available**  
   The application requires a fine-tuned Stable Diffusion model (`fine_tuned_xray_model_full_fixed_v2`). If it’s not already in the project directory, generate it by running:
   ```bash
   python fine_tune_model.py
   ```

2. **Launch the Streamlit App**  
   Start the application with the following command:
   ```bash
   streamlit run app.py
   ```
   - The app will launch in your default web browser at `http://localhost:8501`.![Screenshot 2025-04-19 054728](https://github.com/user-attachments/assets/e82c29e6-9a92-48a5-9a46-50897be17512)
![Screenshot 2025-04-19 064754](https://github.com/user-attachments/assets/3d7a0ad8-4133-4b89-90ad-7d495a76df8e)


### Troubleshooting

If you encounter issues, check the following:
- **Model Not Found**: Verify that the `fine_tuned_xray_model_full_fixed_v2` directory exists in the project folder. Run `python fine_tune_model.py` if it’s missing.
- **Dataset Not Found**: Ensure `preprocessed_data.pkl` is present. Re-run `python preprocess_data.py` if necessary.
- **Dependency Errors**: Confirm all libraries installed correctly by re-running `pip install -r requirements.txt`.

---

## How to Use the Application

Once the app is running:
- **Select a Prompt**: Choose a text description from the dataset or enter your own in the sidebar.
- **Generate an X-ray**: Click the "Generate" button to create a synthetic chest X-ray.
- **View Results**: The app displays the generated X-ray alongside the real one for comparison.

---
