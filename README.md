# 🖼️ Image Dataset Analytics (Cats, Dogs, Birds)

A complete **Data Analytics project** that performs advanced Exploratory Data Analysis (EDA) on an image dataset consisting of **cats, dogs, and birds**.  
This project extracts insights from images using Python, OpenCV, NumPy, Pandas, and Seaborn.

---

## 📌 Features

- Load image datasets from folder structure  
- Extract image properties:  
  - Width, Height  
  - Aspect Ratio  
  - Brightness  
  - RGB color means  
- Detect:
  - Corrupted images  
  - Duplicate images  
- Class-level visualizations:
  - Distribution  
  - Resolution trends  
  - Brightness  
  - Aspect ratio  
  - Color patterns  
- Sample image preview per class  
- CSV export for BI/Visualization tools

---

## 📁 Project Structure
image-dataset-analytics/
│
├── data/
│   └── train/
│       ├── cats/
│       ├── dogs/
│       └── birds/
│
├── src/
│   └── analysis.py
│
├── notebooks/
├── output/
│   └── image_analysis.csv
│
├── requirements.txt
└── README.md
---

## 🚀 How to Run the Project

### 1️⃣ Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the analysis
python src/analysis.py
