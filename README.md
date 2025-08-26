# 🎙️ Speech Emotion Recognition (SER) using BiGRU

## 📌 Overview
This project implements a **Speech Emotion Recognition (SER)** system using a **custom dataset (AVV_dataset)** built by combining samples from **TESS, RAVDESS, SAVEE, EMOV-DB, and CREMA-D**.

The pipeline extracts acoustic and statistical features from speech signals and trains a **Bidirectional GRU (BiGRU)** model to classify emotions into 8 categories:

- 😨 Fear
- 😃 Happy
- 😢 Sad
- 😡 Anger
- 🤢 Disgust
- 😐 Neutral
- 😲 Surprise
- 🌀 Others

---

## ⚙️ Workflow
### 1. Dataset Preparation
- `AVV\_dataset.zip` is structured with speech samples from multiple datasets.
- Each file is labeled based on emotion.

### 2. Feature Extraction
Run the feature extraction script to compute:
- 🎶 MFCC (Mel Frequency Cepstral Coefficients)
- 📊 HT-based Approximate Entropy
- 🔉 Pitch, Energy, Spectral features

➡️ Output: `nppdp.csv` containing extracted features + emotion labels.

### 3. Model Training (BiGRU)
- `nppdp.csv` is used to train a BiGRU model.
- BiGRU captures **temporal dependencies** and improves classification accuracy.

### 4. Prediction
- The trained model predicts emotions from unseen audio samples.

---

## 🛠️ Tech Stack
- **Python 3.12.10**
- **Libraries**:
  - `librosa` → audio feature extraction.
  - `numpy, pandas` → data processing.
  - `scikit-learn` → preprocessing, evaluation.
  - `tensorflow / keras` → BiGRU implementation.
  - `matplotlib, seaborn` → visualization.

---

## 📂 Dataset
The **AVV_dataset** was custom-built to ensure balanced emotion representation.
It merges:
- 🎵 **TESS** – Toronto Emotional Speech Set
- 🎵 **RAVDESS** – Ryerson Audio-Visual Database of Emotional Speech and Song
- 🎵 **SAVEE** – Surrey Audio-Visual Expressed Emotion
- 🎵 **EMOV-DB** – German Emotional Database
- 🎵 **CREMA-D** – Crowd-sourced Emotional Multimodal Actors Dataset

👉 All samples were **standardized** in sampling rate and format, then re-labeled into the **8 emotion categories** defined in this project.

---

## 🖥️ How to Run
1. **Clone the repository**
   ```bash
   git clone https://github.com/sSomaVignesh/Speech-Emotion-Recognition.git
   cd speech-emotion-recognition
   
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **Prepare dataset**
   ```bash
   unzip AVV_dataset.zip -d dataset/

4. **Extract features**
   ```bash
   python feature_extraction.py

5. **Train the model**
   ```bash
   python train_bigru.py

---

## 📈 Results

✅ Achieved **86.6% accuracy** on the test set.

🖼️ Visualizations include:
- Accuracy vs. Epochs
- Loss vs. Epochs
- Confusion Matrix Heatmap

---

## 🚀 Future Work

🔬 Explore **transformer-based architectures** (e.g., wav2vec, HuBERT).

🌍 Extend dataset with **multilingual & real-world noisy samples**.

📱 Deploy as a **real-time mobile/web app**.

🤝 Apply in **healthcare (stress/depression detection)** and **customer support analysis**.
