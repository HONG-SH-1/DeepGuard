# DeepGuard - 딥러닝 기반 실시간 네트워크 침입 탐지 시스템

## 프로젝트 개요
CICIDS2017 데이터셋을 기반으로 2-Tier Cascade 아키텍처를 구현한 실시간 NIDS

- 1차 방어선: MLP Autoencoder (Window=5, relu) - 초고속 필터링
- 2차 방어선: Bi-LSTM Autoencoder (Window=20, tanh) - 정밀 심층 검사

## 기술 스택
- Python 3.10.6
- TensorFlow/Keras, Scikit-learn, Pandas, NumPy
- Streamlit (대시보드)

## 최종 성능
| 공격 유형 | 1차 MLP Recall | 2차 Bi-LSTM Recall |
|---|---|---|
| BruteForce | 0.128 | 0.135 |
| DoS | 0.716 | 0.681 |
| DDoS | 0.836 | 0.905 |
| 평균 | 0.560 | 0.573 |

## 데이터셋
CICIDS2017 (Canadian Institute for Cybersecurity)
- 학습: Monday (BENIGN 100%, 약 238,000개)
- 테스트: Tuesday(BruteForce), Wednesday(DoS), Friday(DDoS)

## 실행 방법

### 환경 설치
pip install -r requirements.txt

### Streamlit 대시보드 실행
streamlit run dashboard/app.py

## 노트북 실행 순서
1. 01_preprocessing.ipynb - 전처리 및 슬라이딩 윈도우 생성
2. 02_tier1_mlp.ipynb - 1차 방어선 학습 (MLP/CNN/GRU 비교 + 튜닝)
3. 03_tier2_bilstm.ipynb - 2차 방어선 학습 (LSTM/Bi-LSTM 비교 + 튜닝)
4. 04_visualization.ipynb - 시각화 생성

## 프로젝트 구조
DeepGuard/
├── dashboard/app.py        # Streamlit 대시보드
├── data/demo_raw_3.csv     # 시연용 데이터 (6,000개)
├── models/                 # 최종 모델 파일
├── notebooks/              # Kaggle 학습 노트북
├── visualizations/         # 시각화 결과물
└── requirements.txt
