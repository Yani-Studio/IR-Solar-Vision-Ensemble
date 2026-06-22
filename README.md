# 🔥 IR-Solar-Vision-Ensemble
> **딥러닝 앙상블 및 TTA 기반 열화상 태양광 패널 결함 탐지 최적화 프로젝트**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458.svg)

## 📌 Project Overview
본 프로젝트는 **열화상 카메라(Thermal Infrared)로 촬영된 태양광 패널(PV) 이미지**를 분석하여 정상(Normal)과 결함(Anomaly)을 높은 정확도로 분류하는 AI 비전 모델 구축 프로젝트입니다. 

단일 모델의 한계를 극복하기 위해 **총 23개의 SOTA(State-of-the-Art) CNN 모델**을 학습시켰으며, **30가지의 앙상블(Ensemble) 기법**을 전수조사하여 최적의 조합을 찾아냈습니다. 더불어 **TTA(Test-Time Augmentation)** 와 **임계치 미세 조정(Threshold Fine-Tuning)** 을 적용하여 분류 성능(Accuracy, F1-Score)을 인간의 인지 한계 수준까지 끌어올렸습니다.

---

## 📊 1. Dataset Overview

본 프로젝트에서 다루는 열화상 태양광 패널 데이터셋은 다양한 종류의 미세 결함을 포함하고 있습니다. 초기 12개의 다중 클래스(Multi-class)로 구성된 데이터를 `Normal(정상)`과 `Faulty(결함)` 이진 분류(Binary Classification) 문제로 재정의하여 모델 학습의 안정성을 극대화했습니다.

### 🔍 태양광 패널 결함 종류 (Anomaly Classes)
![Defect Samples](visualization/Defect_Samples.png)
> Hot-Spot, Cracking, Soiling, Shadowing 등 다양한 열화상 결함 패턴을 딥러닝이 스스로 학습합니다.

### 📈 데이터 클래스 분포 (Class Distribution)
![Class Distribution](visualization/Class_Distribution.png)
> 데이터 불균형(Class Imbalance) 문제를 고려하여 모델 학습 시 가중치 및 계층적 분할(Stratified Split) 기법을 적용했습니다.

---

## 🚀 2. Methodology & Optimization

단순히 하나의 무거운 모델을 사용하는 대신, 여러 모델의 강점을 결합하는 전략을 채택했습니다.

1. **Model Selection**: `ResNet101V2`, `Xception`, `MobileNetV3`, `EfficientNet`, `DenseNet` 등 총 23개의 다양한 구조를 가진 모델을 1차적으로 병렬 학습시켰습니다.
2. **Brute-Force Ensemble Search**: 2~23개의 모델 조합과 30개의 앙상블 가중치 결합 수학 모델(산술/기하/조화 평균, 엔트로피 가중 등)을 전수 조사했습니다.
3. **Threshold Fine-Tuning**: 0.01 단위로 결정 경계(Decision Boundary)를 튜닝하여 False Positive와 False Negative의 밸런스를 맞췄습니다.
4. **Test-Time Augmentation (TTA)**: 추론 과정에서 이미지를 회전, 반전시켜 다각도로 검토하는 TTA 기법을 추가해 모호한 엣지 케이스(Edge Cases)에 대한 강건성을 확보했습니다.

---

## 🏆 3. Evaluation & Results

### 📈 결합 모델 수에 따른 최고 성능 추이
![Accuracy Trend](visualization/Accuracy_Trend.png)
> 모델을 단독으로 사용할 때보다 서로 다른 구조를 결합할수록 성능이 상승하며, **Top-5 모델(ResNet101V2 + ResNet101 + Xception + MobileNetV3Small + DenseNet121)** 에 **기하평균(Geometric Mean)** 앙상블을 적용했을 때 최고의 효율을 보여줍니다.

### 🌌 모델 예측 결과 t-SNE 시각화
![t-SNE Clustering](visualization/t_SNE_Clustering.png)
> 23개 모델의 다차원 확률 벡터를 2차원 공간으로 축소(t-SNE)한 결과, 정상(파란색)과 결함(빨간색)이 명확한 군집을 형성하는 것을 확인할 수 있습니다. 경계면의 혼재 구역은 데이터 자체의 모호성(라벨링 노이즈)을 의미합니다.

---

## 🎯 4. Final Optimization (임계치 튜닝 & 혼동 행렬)

### ⚖️ 결정 임계치 튜닝 (Threshold Tuning)
![Threshold Tuning](visualization/Threshold_Tuning.png)
> 기본 임계치인 0.50에 얽매이지 않고, F1-Score와 Accuracy를 동시에 모니터링하여 **가장 이상적인 커트라인(최적 임계치)** 을 찾아냈습니다.

### ✅ 혼동 행렬 성능 비교 (Single vs Ensemble)
![Confusion Matrix](visualization/Confusion_Matrix.png)
> **(왼쪽) ResNet101V2 단일 모델 ➔ (오른쪽) 최적화 앙상블 모델**
>
> 앙상블 및 최적화를 거치면서 억울하게 결함으로 판정받는 오답(False Positive)과 놓치는 결함(False Negative)이 눈에 띄게 줄어들었으며, 전체 분류 정확도가 극대화되었습니다.
