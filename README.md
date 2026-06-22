# 🔥 IR-Solar-Vision-Ensemble
> **딥러닝 앙상블 및 TTA 기반 열화상 태양광 패널 결함 탐지 최적화 프로젝트**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458.svg)

## 🌟 1. Explainable AI (XAI) 기반 정밀 이상 탐지
본 프로젝트의 앙상블 모델은 단순한 정상/결함 분류를 넘어, **결함의 위치를 시각적으로 완벽하게 짚어내는 설명 가능한 AI(XAI)** 능력을 갖추고 있습니다. 

![Ensemble XAI](visualization/xai.png)
> **12개 결함 유형별 앙상블 Grad-CAM 시각화**
> 정상(No-Anomaly) 데이터는 오탐지 없이 깔끔하게 통과시키고, Cell, Cracking, Hot-Spot 등 다양한 형태의 결함은 픽셀 단위로 정확하게 짚어냅니다. 단일 모델의 얕은 판단을 넘어, 최상위 5개의 SOTA 모델이 기하평균(Geometric Mean)으로 합의한 가장 신뢰도 높은 이상 탐지 결과입니다.

---

## 📈 2. Performance & Optimization (성능 향상 증명)
단일 모델(ResNet101V2)에서 머물지 않고 앙상블과 최적화 튜닝을 거치며 성능이 비약적으로 상승했습니다.

### ✅ 혼동 행렬(Confusion Matrix) 성능 비교
![Confusion Matrix](visualization/Confusion_Matrix.png)
> **(왼쪽) 최고 성능의 단일 모델 ➔ (오른쪽) 최종 최적화 앙상블 모델**
> 앙상블 및 최적화를 거치면서 억울하게 결함으로 판정받는 오답(False Positive)과, 결함을 놓치는 치명적 실수(False Negative)가 눈에 띄게 줄어들었습니다. 이를 통해 전체 분류 정확도와 산업 현장에서의 신뢰도를 극대화했습니다.

### ⚖️ 결정 임계치 튜닝 (Threshold Tuning)
![Threshold Tuning](visualization/Threshold_Tuning.png)
> 기본 임계치인 0.50에 얽매이지 않고, F1-Score와 Accuracy를 동시에 모니터링하여 **가장 이상적인 커트라인(최적 임계치)** 을 찾아내어 탐지 능력을 극한으로 끌어올렸습니다.

---

## 📌 3. Project Overview
본 프로젝트는 **열화상 카메라(Thermal Infrared)로 촬영된 태양광 패널(PV) 이미지**를 분석하여 정상(Normal)과 결함(Anomaly)을 높은 정확도로 분류하는 AI 비전 모델 구축 프로젝트입니다. 

단일 모델의 한계를 극복하기 위해 **총 23개의 SOTA(State-of-the-Art) CNN 모델**을 학습시켰으며, **30가지의 앙상블(Ensemble) 기법**을 전수조사하여 최적의 조합을 찾아냈습니다. 더불어 **TTA(Test-Time Augmentation)** 를 적용하여 분류 성능(Accuracy, F1-Score)을 인간의 인지 한계 수준까지 끌어올렸습니다.

---

## 📊 4. Dataset Overview
초기 12개의 다중 클래스(Multi-class)로 구성된 데이터를 `Normal(정상)`과 `Faulty(결함)` 이진 분류(Binary Classification) 문제로 재정의하여 모델 학습의 안정성을 극대화했습니다.

### 🔍 태양광 패널 결함 종류 (Anomaly Classes)
![Defect Samples](visualization/Defect_Samples.png)
> Hot-Spot, Cracking, Soiling, Shadowing 등 다양한 열화상 결함 패턴을 딥러닝이 스스로 학습합니다.

### 📈 데이터 클래스 분포 (Class Distribution)
![Class Distribution](visualization/Class_Distribution.png)
> 데이터 불균형(Class Imbalance) 문제를 고려하여 모델 학습 시 가중치 및 계층적 분할(Stratified Split) 기법을 적용했습니다.

---

## 🚀 5. Methodology

1. **Model Selection**: `ResNet`, `Xception`, `MobileNet`, `EfficientNet`, `DenseNet` 등 23개의 모델을 병렬 학습시켰습니다.
2. **Brute-Force Ensemble Search**: 2~23개의 모델 조합과 산술/기하/조화 평균 등 30개의 앙상블 결합 수학 모델을 전수 조사했습니다.
3. **Test-Time Augmentation (TTA)**: 추론 과정에서 이미지를 회전, 반전시켜 다각도로 검토하는 기법을 추가해 모호한 엣지 케이스에 대한 강건성을 확보했습니다.

### 🌌 모델 예측 결과 t-SNE 시각화
![t-SNE Clustering](visualization/t_SNE_Clustering.png)
> 23개 모델의 다차원 확률 벡터를 2차원 공간으로 축소(t-SNE)한 결과, 정상(파란색)과 결함(빨간색)이 명확한 군집을 형성하는 것을 확인했습니다.

---

## 🏆 6. Final Model Leaderboard (단일 모델 성능)
본 프로젝트에서 앙상블의 훌륭한 재료가 되어준 **총 23개 개별 모델들의 최종 리더보드**입니다. 

| 순위 | Model | Accuracy | F1-Score | Resolution | Version |
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

> **💡 핵심 인사이트**: 단일 최고 모델(ResNet101V2)도 강력하지만, 상위권에 랭크된 서로 다른 구조의 모델(ResNet, Xception, MobileNet, DenseNet)들을 결합하여 **단일 모델의 성능적 한계를 완벽히 돌파**했습니다.
