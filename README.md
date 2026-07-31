# 🔋 Deep RL & LSTM 기반 동적 환경에서의 배터리 SOC 추정

> **Pontificia Universidade Católica do Paraná (PUCPR)**  
> **저자:** Gabriel Vidal Schneider, Julian Ballestros Ramos, Vitor Solieri do Vale, Yejin Chung  
> **분야:** Machine Learning / Reinforcement Learning / Battery Management Systems (BMS)

---

## 📌 1. 프로젝트 개요 (Project Overview)

본 연구는 동적 부하 환경에서 노화된 리튬 이온 배터리의 **잔존 용량(State of Charge, SOC)**을 정확하게 추정하기 위한 **강화학습(Reinforcement Learning, RL)** 기술의 적용 가능성을 탐구합니다[cite: 1].

기존의 전통적인 SOC 추정 방식(예: 개로 전압법 - OCV, 쿨롱 카운팅 - Coulomb Counting)은 배터리 내부의 전기화학적 평형을 맞추기 위해 긴 휴식 시간이 필요하며, 실제 운행 중 발생하는 동적이고 노이즈가 많은 조건에서 한계를 보입니다[cite: 1]. 

본 프로젝트에서는 **Long Short-Term Memory (LSTM)** 기반의 시계열 베이스라인 모델과 커스텀 **Gymnasium** 환경에서 **Proximal Policy Optimization (PPO)** 알고리즘으로 학습된 강화학습 에이전트를 구축하고 성능을 비교 분석했습니다[cite: 1].

### 🌟 핵심 하이라이트 (Key Highlights)
- **실제 노화 배터리 데이터셋 활용:** 2차 수명(SoH 60%~85%) 및 3차 수명(SoH 22%~45%)에 달하는 총 6개의 노화 배터리 셀 데이터 기반 검증[cite: 1].
- **커스텀 강화학습 환경 구축:** `Gymnasium` 및 `PPO` 알고리즘을 활용한 연속 제어(Continuous Control) 기반 문제 정형화[cite: 1].
- **4.2배 빠른 학습 속도:** Stacked LSTM 모델(~2시간 30분) 대비 PPO 강화학습 모델(**~35분**)의 학습 속도 대폭 개선[cite: 1].
- **특정 셀 맞춤형 정밀도:** 심각하게 노화된 3차 수명 배터리 셀에서 **최저 0.0985 RMSE** 달성[cite: 1].

---

## 📊 데이터셋 및 전처리 (Dataset & Preprocessing)

본 연구에서는 다양한 노화 단계의 연속 방전 데이터를 포함하고 있는 **IEEE DataPort Experimental Dataset of Retired Batteries** [1]를 활용하였습니다[cite: 1].

### 🪫 배터리 셀 프로필 (Battery Cell Profiles)

| 셀 ID | 수명 단계 | 정격 용량 (Capacity) | 건강 상태 (SoH) |
| :---: | :---: | :---: | :---: |
| **Cell 001** | 2nd Life (재사용) | 1870 mAh | 85% |
| **Cell 002** | 2nd Life (재사용) | 1804 mAh | 82% |
| **Cell 003** | 2nd Life (재사용) | 1320 mAh | 60% |
| **Cell 004** | 3rd Life (재활용 전) | 990 mAh | 45% |
| **Cell 005** | 3rd Life (재활용 전) | 530 mAh | 24% |
| **Cell 006** | 3rd Life (재활용 전) | 490 mAh | 22% |

> 각 셀은 정전류 조건에서 20회의 방전 사이클을 거쳤으며, 4사이클마다 저항이 0.5Ω씩 증가하도록 설정되었습니다[cite: 1]. 수집된 피처: **전압(V), 전류(A), 전력(W), 저항(Ω), 온도(°C)**[cite: 1].

### 🔄 전처리 및 쿨롱 카운팅 (Coulomb Counting)

원본 데이터셋에 Ground-Truth SOC가 명시적으로 제공되지 않았기 때문에, **쿨롱 카운팅(Coulomb Counting)** 적분법을 수행한 후 **Min-Max Normalization**을 적용하여 각 방전 사이클의 SOC를 산출하였습니다[cite: 1].

$$\Delta SOC = \frac{-I \cdot \Delta t}{C}$$

$$SOC = 1 - \sum \Delta SOC$$

$$SOC_{\text{norm}} = \frac{SOC - SOC_{\text{min}}}{SOC_{\text{max}} - SOC_{\text{min}}} \in [0, 1]$$

---

## 🛠️ 모델 아키텍처 및 파이프라인 (Model Architectures)

### 1️⃣ 베이스라인 모델: Stacked LSTM
- **입력 피처:** 전압, 전류, 전력, 저항, 온도 (Min-Max Scaling 적용)[cite: 1]
- **시퀀스 창 크기:** 슬라이딩 윈도우 크기 $T = 20$[cite: 1]
- **아키텍처:** 2단 Stacked LSTM + Dropout 레이어 + Dense 출력 레이어[cite: 1]
- **손실 함수:** 평균 제곱 오차 (MSE)[cite: 1]

### 2️⃣ 제안 모델: 강화학습 (PPO)
- **환경 (Environment):** 커스텀 `Gymnasium` 배터리 환경[cite: 1]
- **상태 공간 (Observation Space):** $[V, I, P, R, T]$[cite: 1]
- **행동 공간 (Action Space):** $[0, 1]$ 범위의 연속적인 SOC 예측값[cite: 1]
- **보상 함수 (Reward Function):** 예측 SOC($\widehat{SOC}$)와 실제 $SOC$ 간 제곱 오차 감점: $R = -(\widehat{SOC} - SOC)^2$[cite: 1]
- **알고리즘:** Proximal Policy Optimization (PPO), 총 **200,000 Step** 학습[cite: 1]

---

## 📈 실험 결과 (Experimental Results)

### 🧪 모델 성능 비교 (Model Performance)

#### 🔹 Stacked LSTM (이종 배터리 간 일반화 성능)
학습에 사용되지 않은 배터리에 대한 일반화 성능을 평가하기 위해 교차 분할 실험 수행[cite: 1]:

| 실험 시나리오 | 학습 셀 개수 | 검증 셀 개수 | RMSE |
| :--- | :---: | :---: | :---: |
| **시나리오 1** | 1개 셀 | 5개 셀 | **0.181** |
| **시나리오 2** | 3개 셀 | 3개 셀 | **0.185** |
| **시나리오 3** | 5개 셀 | 1개 셀 | **0.189** |

*LSTM은 새로운 배터리 셀에 대해서도 일관된 0.181 ~ 0.189 수준의 RMSE를 유지하며 우수한 일반화 성능을 보임[cite: 1].*

#### 🔹 PPO Reinforcement Learning (셀 맞춤형 적응 성능)
각 셀의 방전 사이클 중 앞선 50%로 학습하고, 후반 50% 사이클로 검증 수행[cite: 1]:

| 대상 배터리 셀 | 수명 단계 | 건강 상태 (SoH) | RMSE |
| :--- | :---: | :---: | :---: |
| **Cell 001** | 2nd Life | 85% | 0.2553 |
| **Cell 002** | 2nd Life | 82% | 0.2746 |
| **Cell 003** | 2nd Life | 60% | 0.1921 |
| **Cell 004** | 3rd Life | 45% | 0.1204 |
| **Cell 005** | 3rd Life | 24% | 0.1359 |
| **Cell 006** | 3rd Life | 22% | **0.0985** |

*PPO 모델은 심각하게 열화된 3차 수명 배터리(Cell 004~006)일수록 향상된 추정 정밀도를 기록함 (**최저 0.0985 RMSE**)[cite: 1].*

---

## ⚡ 연산 효율성 비교 (Computational Cost)

| 모델 | 전체 학습 및 검증 소요 시간 | 주요 장점 |
| :--- | :---: | :--- |
| **Stacked LSTM** | 약 2시간 30분 | 미학습 배터리에 대한 높은 교차 일반화(Generalization) 능력[cite: 1] |
| **RL (PPO)** | **약 35분** | **4.2배 빠른 학습**, 개별 셀 특성에 맞춘 뛰어난 적응력(Adaptability)[cite: 1] |

---

## 💡 결론 및 인사이트 (Conclusions & Key Takeaways)

1. **일반화(Generalization) vs 적응성(Adaptability)의 트레이드오프:**
   - **LSTM**은 이종 배터리 팩 전반에 범용적으로 적용할 수 있는 글로벌 모델로 적합합니다[cite: 1].
   - **PPO 강화학습**은 특정 배터리 셀의 노화 상태에 맞춘 파인튜닝 및 온디바이스(On-device) 적응형 BMS에 매우 효과적입니다[cite: 1].
2. **연산 효율성 증대:** 강화학습 접근 방식은 학습 시간을 80% 이상 단축시켜 컴퓨팅 자원이 제한된 실시간 시스템 적용 가능성을 제시합니다[cite: 1].
3. **휴식 시간 배제:** 두 데이터 기반 접근법 모두 배터리 안정화를 위한 휴식 시간 없이 동적 운행 환경에서 SOC를 성공적으로 추정했습니다[cite: 1].

---

## 📚 참고 문헌 (References)

1. J. D. Gotz, N. Mendes, A. de Souza Britto Junior, J. R. Galvão, F. C. Corrêa, A. A. Badin, *"Experimental dataset of retired batteries (first-life, second-life and third-life) discharged at constant load rate"*, IEEE DataPort, 2026. DOI: [10.21227/mm0k-2w93](https://dx.doi.org/10.21227/mm0k-2w93)[cite: 1].
2. V. Selvaraj, I. Vairavasundaram, *"A comprehensive review of state of charge estimation in lithium-ion batteries used in electric vehicles"*, Journal of Energy Storage, 2023[cite: 1].
3. K. Luo, X. Chen, H. Zheng, Z. Shi, *"A review of deep learning approach to predicting the state of health and state of charge of lithium-ion batteries"*, Journal of Energy Chemistry, 2022[cite: 1].
4. R. S. Sutton, A. G. Barto, *"Reinforcement Learning: An Introduction"*, MIT Press, 2018[cite: 1].

---

## 📬 저자 정보 (Contact & Citation)

- **소속:** Pontificia Universidade Católica do Paraná (PUCPR), Curitiba, Brazil[cite: 1]  
- **저자:** Gabriel Vidal Schneider, Julian Ballestros Ramos, Vitor Solieri do Vale, Yejin Chung[cite: 1]
