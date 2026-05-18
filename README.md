# NeoScan AI — Advanced Neonatal Screening 🍼

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge&logo=google-chrome&logoColor=white)](https://bytecraft17.github.io/NeoScan-Website/Frontend/)
[![AI Research](https://img.shields.io/badge/AI-Research-blue?style=for-the-badge&logo=github&logoColor=white)](https://github.com/bytecraft17/Neoscan-AI-)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)

**NeoScan AI** is a state-of-the-art diagnostic platform designed for early neonatal screening. Utilizing advanced Deep Learning architectures, the system analyzes images of newborns to detect 15+ medical conditions across four specialized modules, providing immediate risk assessment and clinical advice.

**🌍 Live Website:** [https://bytecraft17.github.io/NeoScan-Website/Frontend/](https://bytecraft17.github.io/NeoScan-Website/Frontend/)

---

## 🚀 Key Features

- 🧠 **Multi-Module AI**: Specialized neural networks for Skin, Eye, Face, and Body analysis.
- ⚡ **Real-Time Inference**: Powered by FastAPI with support for both Live (Model-based) and Demo (Deterministic) modes.
- 📱 **Responsive Interface**: Modern, mobile-friendly UI for bedside clinical use.
- 🩺 **Clinical Decision Support**: Provides risk levels (Normal, Moderate, Critical) and actionable advice for healthcare providers.

---

## 🧩 Diagnostic Modules

### 1. Skin Analysis Module
- **Model**: EfficientNetB3 (PyTorch)
- **Detections**: Jaundice, Cyanosis, Anemia, Normal.
- **Preprocessing**: Gray World White Balance & CLAHE for robust performance under phototherapy/hospital lighting.

### 2. Eye Analysis Module
- **Model**: EfficientNetB4 (TensorFlow/Keras)
- **Detections**: Retinopathy of Prematurity (ROP) Stages 1-3, Cataracts/Leukocoria.
- **Pipeline**: Automated contrast enhancement and Gaussian sharpening for retinal clarity.

### 3. Face Analysis Module (Genetic/Congenital)
- **Model**: Ensemble (VGG16 + Vision Transformer - ViT)
- **Detections**: Down Syndrome, Cleft Lip/Palate, Healthy Features.

### 4. Body Analysis Module
- **Model**: ResNet50 (PyTorch)
- **Detections**: Postural abnormalities, limb defects, and hydrops detection via musculoskeletal symmetry analysis.

---

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3 (Vanilla), JavaScript (ES6+)
- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **AI/ML**: PyTorch, TensorFlow, Keras, OpenCV, TIMM
- **Deployment**: GitHub Pages (Frontend), Local/Cloud (Backend)

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/bytecraft17/NeoScan-Website.git
cd NeoScan-Website
```

### 2. Setup Backend
```bash
cd Backend
pip install -r requirements.txt
```

### 3. Model Weights
The pre-trained model weights (.pth and .keras) are not included in the repository due to their size. 
- **Download Weights**: [Google Drive Folder](https://drive.google.com/drive/folders/1Sj9OCNijpDyL0qnsxFj2BNnAR8N4KSQ6?usp=sharing)
- Place the weights in the `Backend/` directory.

### 4. Run the Application
```bash
# Start the FastAPI server
python main.py
```
The application will be available at `http://127.0.0.1:8000`.

---

## 📂 Project Structure

```text
NeoScan_Website/
├── Backend/
│   ├── main.py              # FastAPI Entry Point & Prediction Logic
│   ├── requirements.txt      # Python Dependencies
│   └── *.pth / *.keras       # AI Model Weights (To be downloaded)
├── Frontend/
│   ├── index.html           # Main Application UI
│   ├── css/                 # Styling & Design System
│   └── js/                  # Frontend Logic & API Integration
└── README.md
```

---

## 👨‍💻 AI Notebooks & Research
For the full training pipelines, datasets, and architectural experiments, visit the research repository:
[Neoscan-AI Research Repo](https://github.com/bytecraft17/Neoscan-AI-)

---
*Disclaimer: NeoScan AI is an assistive screening tool and should not replace professional clinical judgment.*

