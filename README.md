# 🔥 IR-Aero-Solar-Vision-Ensemble

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458.svg)

---
> ⚠️ Copyright (c) 2026 Kang Gyu Min. All rights reserved.
---

<table>
  <tr>
    <td width="40%">
      <img src="visualization/checking.png" width="100%">
    </td>
    <td width="60%">
      <h3>"Can human inspectors manually check tens of thousands of solar panels?"</h3>
      This project is an AI vision model that utilizes Kaggle's <b>'Infrared Solar Modules' dataset</b> to automatically detect critical defects (e.g., hotspots, diode failures, cell cracks) in solar panels occurring in real industrial environments.<br><br>
      By analyzing temperature distribution images captured by <b>Thermal Infrared</b> cameras mounted on drones or inspection equipment, the model achieves a high accuracy of <b>96.2%</b> in identifying internal energy leaks and defects that are completely invisible to the naked eye.
    </td>
  </tr>
</table>

---

## 🎯 1. Final Achievement
To overcome the performance limitations of a single model, the top 5 models with different architectures were combined using the Geometric Mean, achieving an overwhelming defect detection **final test classification accuracy of 96.2%**.

**🏆 [Final Ensemble Optimization Results]**
| Category | Result |
|:---|:---|
| **Best Performance** | **Accuracy: 96.2%** |
| **Final Ensemble Models (Top 5)** | `ResNet101V2`, `ResNet101`, `Xception`, `MobileNetV3Small`, `DenseNet121` |
| **Optimal Combination Method** | Geometric Mean |
| **Threshold Tuning Value** | Tuned Optimal Threshold (from 0.54) |

---

## 🌟 2. Precision Anomaly Detection based on Explainable AI (XAI)
Beyond simple normal/defect classification, the ensemble model in this project possesses **Explainable AI (XAI) capabilities that visually pinpoint the exact location of defects**.

![Ensemble XAI](visualization/xai.png)
> **Ensemble Grad-CAM Visualization by 12 Defect Types**
> Normal (No-Anomaly) data passes cleanly without false positives, while various defect types such as Cells, Cracking, and Hot-Spots are accurately pinpointed at the pixel level. This is the highly reliable anomaly detection result agreed upon by the top 5 SOTA models.

---

## 📈 3. Performance Improvement
The performance increased dramatically by going through the ensemble pipeline rather than staying with a single model (ResNet101V2).

### ✅ Confusion Matrix Performance Comparison
![Confusion Matrix](visualization/Confusion_Matrix.png)
> **(Left) Best Performing Single Model ➔ (Right) Final Optimized Ensemble Model (96.2%)**
> Through the ensemble and optimization processes, false positives (wrongly judged as defects) and false negatives (critical misses of actual defects) were significantly reduced. This maximized the overall classification accuracy and reliability in industrial applications.

---

## 📌 4. Project Overview
This project aimed to build an advanced deep learning pipeline that analyzes **images of solar panels (PV) captured with Thermal Infrared cameras** to classify them into Normal and Anomaly with high accuracy.

To achieve this, a total of **23 State-of-the-Art (SOTA) CNN models** were trained. Going beyond simply averaging the top models, we conducted a **brute-force search of 30 mathematical ensemble combination methods** to find the optimal combination. Furthermore, by fine-tuning the threshold, the detection performance was optimized to the level of human cognitive limits.

---

## 📊 5. Dataset Overview

### 🔍 Solar Panel Defect Types (Anomaly Classes)
![Defect Samples](visualization/Defect_Samples.png)
> Deep learning automatically learns various thermal defect patterns such as Hot-Spots, Cracking, Soiling, and Shadowing.

### 📈 Data Distribution & Clustering Analysis
<table width="100%">
  <tr>
    <th width="50%">📊 Class Distribution</th>
    <th width="50%">🌌 Original Data t-SNE Clustering Visualization</th>
  </tr>
  <tr>
    <td align="center"><img src="visualization/Class_Distribution.png"></td>
    <td align="center"><img src="visualization/t_SNE_Clustering.png"></td>
  </tr>
  <tr>
    <td>
      <b>[Bias Prevention & Binary Classification]</b><br>
      To prevent learning bias caused by data imbalance among the original 12 defect types, the data was redefined into <b>two classes: Normal and Faulty</b>, and Stratified Split was applied to maximize learning stability.
    </td>
    <td>
      <b>[Preliminary Analysis of Data Difficulty]</b><br>
      By reducing the high-dimensional original images to 2 dimensions (t-SNE), the distribution and ambiguous boundaries between normal and defect data were analyzed in advance.
    </td>
  </tr>
</table>

---

## 🚀 6. Methodology (Preprocessing & Ensemble Optimization Pipeline)
Beyond simply loading and training models, we built a pipeline that fundamentally blocks data loss and finds the perfect ensemble combination through tens of millions of computations.

### ① Lossless Data Preprocessing & Resolution Optimization
- **Anti-Distortion Padding**: The original 40x24 resolution images were padded (ZeroPadding2D) to 40x40 to prevent shape distortion during resizing.
- **Lossless Channel Adapter**: To input grayscale (1-channel) images into color (3-channel) models, tensor duplication (Lambda) was used instead of Conv2D to fundamentally block initial image loss.
- Images were scaled using Bicubic interpolation to the optimal recommended resolution tailored to each model's architecture (e.g., 224x224), followed by the dedicated `preprocess_input`.

### ② 3-Stage Progressive Fine-Tuning
- **[Stage 1] Classifier Warm-up**: The pre-trained backbone network was completely frozen, and only the top classifier was trained first to prevent the destruction of initial weights.
- **[Stage 2] Fine-tuning Top 50 Layers**: The top 50 layers of the model were unfrozen and fine-tuned with a low learning rate. (BatchNormalization remained frozen).
- **[Stage 3] Final Optimization**: Up to 75 top layers were additionally unfrozen, and the `CosineDecay` learning rate scheduler was applied to settle into the Global Minimum.

### ③ Data Augmentation & Early Stopping
- Model generalization was enhanced through image augmentation (rotation, flipping, etc.) using `ImageDataGenerator`, and `EarlyStopping(patience=12)` was applied to all 23 models to secure the best weights and prevent overfitting.

### ④ Brute-Force Ensemble Search & Threshold Tuning
Based on the top 23 models, a total of **30 mathematical ensemble combination methods** were cross-applied to all possible combinations ranging from 2 to 23 models, performing tens of millions of computations.

![Accuracy Trend](visualization/Accuracy_Trend.png)
> **[STEP 1] Peak Performance Trend by Number of Combined Models** 
> The brute-force search proved that the highest synergy was achieved when the top 5 models were combined using the **'Geometric Mean'**.

![Threshold Tuning](visualization/Threshold_Tuning.png)
> **[STEP 2] Threshold Fine-Tuning**
> After confirming the ensemble combination, rather than sticking to the default threshold of 0.50, we simultaneously monitored the F1-Score and Accuracy to finely tune the **most ideal cut-off line (optimal threshold)**, pushing the detection capability to its absolute limit.

<br>

**[30 Ensemble Combination Methods Used for Search]**
| No | Ensemble Method | No | Ensemble Method |
|:---:|:---|:---:|:---|
| 1 | Hard Voting | 16 | Max Probability |
| 2 | Arithmetic Mean | 17 | Min Probability |
| 3 | Accuracy Weighted | 18 | Min-Max Mean |
| 4 | F1 Weighted | 19 | Interquartile Range (IQR) Mean |
| 5 | Exp Rank Weighted | 20 | Rank Mean |
| 6 | Softmax Weighted | 21 | Weighted Rank Mean |
| 7 | Geometric Mean | 22 | Reciprocal Rank Fusion (RRF, k=60) |
| 8 | Harmonic Mean | 23 | Entropy Weighted (Shannon) |
| 9 | Median | 24 | Neg-Entropy Weighted |
| 10 | Trimmed Mean | 25 | KL Divergence Weighted |
| 11 | Winsorized Mean | 26 | Logit Mean |
| 12 | Power (p=0.5) | 27 | Confidence Weighted |
| 13 | Power (p=2) | 28 | Product Rule |
| 14 | Power (p=3) | 29 | Sharp Weighted |
| 15 | Lehmer (p=2) | 30 | Inverse Error Weighted |

---

## 🏆 7. Final Model Leaderboard (Single Model Performance)
This is the final leaderboard of all 23 individual models that served as excellent ingredients for the ensemble in this project.

| Rank | Model | Accuracy | F1-Score | Resolution | Version |
|:---:|:---|:---:|:---:|:---:|:---|
| 🥇 1 | **ResNet101V2** | 0.94775 | 0.947739 | 224 | Optimized |
| 🥈 2 | **ResNet101** | 0.94750 | 0.947463 | 224 | Optimized |
| 🥉 3 | **Xception** | 0.94725 | 0.947244 | 299 | Optimized |
| 4 | **ResNet50V2** | 0.94425 | 0.944244 | 224 | Optimized |
| 5 | **MobileNetV3Small** | 0.94400 | 0.943989 | 224 | Optimized |
| 6 | MobileNetV3Large | 0.94100 | 0.940987 | 224 | Optimized |
| 7 | ResNet50 | 0.93975 | 0.939734 | 224 | Optimized |
| 8 | InceptionResNetV2| 0.93925 | 0.939223 | 299 | Optimized |
| 9 | MobileNetV1 | 0.93675 | 0.936739 | 224 | Optimized |
| 10 | MobileNetV2 | 0.93350 | 0.933485 | 224 | Optimized |
| 11 | **DenseNet121** | 0.93325 | 0.933227 | 224 | Optimized |
| 12 | InceptionV3 | 0.93175 | 0.931711 | 299 | Optimized |
| 13 | EfficientNetV2S | 0.93075 | 0.930651 | 384 | Optimized |
| 14 | EfficientNetV2B0 | 0.92625 | 0.926145 | 224 | Optimized |
| 15 | EfficientNetV2B3 | 0.92475 | 0.924589 | 300 | Optimized |
| 16 | EfficientNetV2B1 | 0.92425 | 0.924153 | 240 | Optimized |
| 17 | EfficientNetV2B2 | 0.92300 | 0.922954 | 260 | Optimized |
| 18 | NASNetMobile | 0.90725 | 0.907191 | 224 | Optimized |
| 19 | EfficientNetB3 | 0.87200 | 0.870330 | 300 | Optimized |
| 20 | EfficientNetB4 | 0.85900 | 0.856957 | 380 | Optimized |
| 21 | EfficientNetB0 | 0.82050 | 0.815287 | 224 | Optimized |
| 22 | EfficientNetB2 | 0.81825 | 0.814965 | 260 | Optimized |
| 23 | EfficientNetB1 | 0.78400 | 0.774232 | 240 | Optimized |

> **💡 Key Insight**: While the performance of the single best model (ResNet101V2) is impressive, combining the top 5 models—each with different strengths—using the **Geometric Mean** ensemble (discovered through a brute-force search of 30 mathematical methods) **perfectly broke through the limitations of a single model (0.947), achieving 96.2%**.

---

## 📚 8. References
- **Data Source**: [Kaggle - Infrared Solar Modules](https://www.kaggle.com/datasets/marcosgabriel/infrared-solar-modules/data)
- **Methodology Reference**: [Kaggle - Solar Modules Fault Detection with CNN (PyTorch)](https://www.kaggle.com/code/aliakbaryaghoubi/solar-modules-fault-detection-with-cnn-pytorch)
- **Industrial Image Source**: [MapperX - How to inspect solar power plants with drone](https://mapperx.com/en/pv-panel-inspection-software/how-to-inspect-solar-power-plants-with-drone/)
