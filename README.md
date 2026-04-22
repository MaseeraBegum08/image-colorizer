"# image-colorizer" 
# 🎨 AI Image Colorizer

A Python app that automatically colorizes black & white images using deep learning.

## 🚀 Features
- Converts black & white images to color
- Uses pre-trained deep learning model
- Simple GUI interface

## ⚙️ Installation

pip install -r requirements.txt


## ▶️ Run the App

streamlit run app.py


## 📥 Download AI Model File
The model file is too large for GitHub (125MB).  
Download it here and place it in the project folder:

👉 [Download colorization_release_v2.caffemodel - Google Drive](https://drive.google.com/file/d/1MiVltDxd_QM8Koixcxg5ylX4s8YtXEuH/view?usp=drive_link)

## 📂 Project Structure
- app.py - Main Streamlit app
- PySimpleGUI_Colorizer.py - GUI code
- colorization_deploy_v2.prototxt - Model config
- pts_in_hull.npy - Color reference data
- colorization_release_v2.caffemodel - AI model (download separately)
