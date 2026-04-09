# dashboard/app.py
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import time
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR              = Path(__file__).parent.parent
MODELS_DIR            = BASE_DIR / 'models'
DATA_DIR              = BASE_DIR / 'data'

MLP_MODEL_PATH        = MODELS_DIR / 'TUN2_u64_p90_nodrop.keras'
MLP_THRESHOLD_PATH    = MODELS_DIR / 'TUN2_u64_p90_nodrop_threshold.npy'
BILSTM_MODEL_PATH     = MODELS_DIR / 'bilstm_tanh_base.keras'
BILSTM_THRESHOLD_PATH = MODELS_DIR / 'bilstm_tanh_base_threshold.npy'
SCALER_PATH           = MODELS_DIR / 'scaler.pkl'
DEMO_RAW_PATH         = DATA_DIR   / 'demo_raw_3.csv'

# ── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepGuard - 실시간 침입 탐지",
    page_icon="",
    layout="wide"
)

# ── 모델 + 스케일러 로드 (캐싱) ───────────────────────────────
@st.cache_resource
def load_models():
    mlp_model        = tf.keras.models.load_model(MLP_MODEL_PATH)
    mlp_threshold    = float(np.load(MLP_THRESHOLD_PATH))
    bilstm_model     = tf.keras.models.load_model(BILSTM_MODEL_PATH)
    bilstm_threshold = float(np.load(BILSTM_THRESHOLD_PATH))
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return mlp_model, mlp_threshold, bilstm_model, bilstm_threshold, scaler

# ── 동기화 슬라이딩 윈도우 생성 ───────────────────────────────
def create_synchronized_windows(X_scaled, labels, attack_types,
                                 window_small=5, window_large=20):
    """
    같은 시점 t에서 W5/W20 동시 생성
    → X_w5[i] ↔ X_w20[i] 완벽 매핑
    실무 NIDS 패킷 버퍼링 메커니즘 구현
    """
    X_w5_list   = []
    X_w20_list  = []
    labels_list = []
    atypes_list = []

    for t in range(window_large, len(X_scaled)):
        X_w5_list.append(X_scaled[t-window_small:t])
        X_w20_list.append(X_scaled[t-window_large:t])
        if labels is not None:
            labels_list.append(labels[t])
        if attack_types is not None:
            atypes_list.append(attack_types[t])

    return (
        np.array(X_w5_list,  dtype=np.float32),
        np.array(X_w20_list, dtype=np.float32),
        np.array(labels_list)      if labels_list  else None,
        np.array(atypes_list)      if atypes_list  else None
    )

# ── 전처리 + 윈도우 생성 함수 ─────────────────────────────────
def preprocess_and_window(df, scaler, feature_names):
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        return None, None, None, None, \
               f"누락된 컬럼 {len(missing)}개: {list(missing[:5])}{'...' if len(missing) > 5 else ''}"

    X        = df[feature_names].values.astype(np.float32)
    X_scaled = scaler.transform(X)

    labels       = df['label'].values       if 'label'       in df.columns else None
    attack_types = df['attack_type'].values if 'attack_type' in df.columns else None

    X_w5, X_w20, labels_win, atypes_win = create_synchronized_windows(
        X_scaled, labels, attack_types
    )
    return X_w5, X_w20, labels_win, atypes_win, None

# ── 2-Tier 배치 추론 함수 ──────────────────────────────────────
def predict_2tier_batch(X_w5, X_w20,
                        mlp_model, mlp_threshold,
                        bilstm_model, bilstm_threshold,
                        status_ph, progress_ph):
    total = len(X_w5)

    # ── 1차 방어선: MLP ───────────────────────────────────────
    status_ph.info("[1차] 방어선 (MLP) 분석 중...")
    progress_ph.progress(10)

    recon_mlp  = mlp_model.predict(X_w5, verbose=0, batch_size=256)
    mse_mlp    = np.mean(np.power(X_w5 - recon_mlp, 2), axis=(1, 2))
    tier1_flag = mse_mlp > mlp_threshold

    status_ph.success(f" 1차 방어선 완료 | ⚠️ 의심 트래픽: {tier1_flag.sum()}개")
    progress_ph.progress(40)
    time.sleep(0.3)

    # ── 2차 방어선: Bi-LSTM ───────────────────────────────────
    status_ph.info("[2차] 방어선 (Bi-LSTM) 분석 중...")
    progress_ph.progress(50)

    suspect_idx = np.where(tier1_flag)[0]
    tier2_mse   = np.full(total, np.nan)

    if len(suspect_idx) > 0:
        X_w20_suspect = X_w20[suspect_idx]
        recon_w20     = bilstm_model.predict(X_w20_suspect, verbose=0, batch_size=256)
        mse_w20       = np.mean(np.power(X_w20_suspect - recon_w20, 2), axis=(1, 2))
        tier2_mse[suspect_idx] = mse_w20

    progress_ph.progress(80)
    status_ph.success(f" 2차 방어선 완료 | 검사 샘플: {len(suspect_idx)}개")
    time.sleep(0.3)

    # ── 결과 취합 ─────────────────────────────────────────────
    status_ph.info("결과 생성 중...")
    progress_ph.progress(90)

    tier2_flag = np.zeros(total, dtype=bool)
    if len(suspect_idx) > 0:
        valid_mask = ~np.isnan(tier2_mse[suspect_idx])
        valid_idx  = suspect_idx[valid_mask]
        tier2_flag[valid_idx] = tier2_mse[valid_idx] > bilstm_threshold

    result_df = pd.DataFrame({
        '샘플'    : np.arange(total),
        '1차MSE'  : mse_mlp,
        '1차결과' : np.where(tier1_flag, '⚠️ 의심', '✅ 정상'),
        '2차MSE'  : tier2_mse,
        '2차결과' : np.where(
                        ~tier1_flag, '미검사',
                        np.where(tier2_flag, '🚨 공격', '✅ 정상')
                    ),
        '최종판정': np.where(tier2_flag, '🚨 공격', '✅ 정상')
    })

    progress_ph.progress(100)
    status_ph.success("분석 완료!")
    time.sleep(0.3)
    progress_ph.empty()

    return result_df

# ── UI 시작 ───────────────────────────────────────────────────
st.title("DeepGuard")
st.subheader("딥러닝 기반 실시간 네트워크 침입 탐지 시스템")
st.markdown("---")

with st.spinner("모델 로딩 중... 잠시만 기다려주세요."):
    mlp_model, mlp_threshold, bilstm_model, bilstm_threshold, scaler = load_models()
FEATURE_NAMES = scaler.feature_names_in_
st.success(" 모델 로드 완료")

# ── 사이드바 ───────────────────────────────────────────────────
st.sidebar.title("시스템 정보")
st.sidebar.markdown("**모델 정보**")
st.sidebar.info(f"""
- 1차 방어선: MLP (TUN2)
- 임계값: {mlp_threshold:.6f}
- 2차 방어선: Bi-LSTM (tanh)
- 임계값: {bilstm_threshold:.6f}
""")

st.sidebar.markdown("**아키텍처**")
st.sidebar.markdown("📥 전체 트래픽 입력")
st.sidebar.markdown("&nbsp;&nbsp;&nbsp;&nbsp;⬇️")
st.sidebar.markdown("🔷 **1차 방어선: MLP (W=5)**")
st.sidebar.markdown("&nbsp;&nbsp;— 초고속 필터링 (5.66s/epoch)")
st.sidebar.markdown("&nbsp;&nbsp;정상 → 통과")
st.sidebar.markdown("&nbsp;&nbsp;의심 → 2차로 전달")
st.sidebar.markdown("&nbsp;&nbsp;&nbsp;&nbsp;⬇️ 의심 트래픽만")
st.sidebar.markdown("🔶 **2차 방어선: Bi-LSTM (W=20)**")
st.sidebar.markdown("&nbsp;&nbsp;— 정밀 심층 검사 (120.77s/epoch)")
st.sidebar.markdown("&nbsp;&nbsp;정상 → 통과")
st.sidebar.markdown("&nbsp;&nbsp;공격 → 차단")
st.sidebar.markdown("---")

st.sidebar.markdown("**데이터셋 정보**")
st.sidebar.info("""
- CICIDS2017 기반
- 피처: 78개
- 학습: Monday (BENIGN)
- 테스트: Tue/Wed/Fri
""")

# ── 데이터 입력 ───────────────────────────────────────────────
st.markdown("## 데이터 입력")
tab1, tab2 = st.tabs(["데모 데이터", "CSV 업로드"])

with tab1:
    st.markdown("""
    **데모 데이터 구성 (CICIDS2017)**
    | 공격 유형 | 원본 샘플 | 윈도우 생성 후 |
    |---|---|---|
    | BruteForce (Tuesday) | 2,000개 | ~1,980개 |
    | DoS (Wednesday) | 2,000개 | ~1,980개 |
    | DDoS (Friday) | 2,000개 | ~1,980개 |

    ※ 동기화 슬라이딩 윈도우 적용으로 앞 20개 샘플 제외
    """)

    if st.button("데모 데이터로 분석", use_container_width=True, type="primary"):
        with st.spinner("전처리 중... 슬라이딩 윈도우 생성 중..."):
            df_demo = pd.read_csv(DEMO_RAW_PATH)
            X_w5, X_w20, labels, attack_types, error = preprocess_and_window(
                df_demo, scaler, FEATURE_NAMES
            )

        if error:
            st.error(error)
        else:
            st.session_state['X_w5']         = X_w5
            st.session_state['X_w20']        = X_w20
            st.session_state['labels']       = labels
            st.session_state['attack_types'] = attack_types
            st.session_state['ready']        = True
            st.success(f" 데모 데이터 로드 완료: {len(X_w5)}개 샘플 | 동기화 윈도우 생성 ")

with tab2:
    st.info("""
    **CSV 업로드 조건**
    - CICIDS2017 기준 78개 피처 포함
    - 스케일링 불필요 (앱 내부에서 자동 처리)
    - 선택 컬럼: `label`, `attack_type` (있으면 탐지율 분석 가능)
    """)
    with st.expander("필요한 78개 피처 목록 보기"):
        feat_df = pd.DataFrame(
            FEATURE_NAMES.reshape(-1, 3),
            columns=['피처1', '피처2', '피처3']
        )
        st.dataframe(feat_df, use_container_width=True)

    uploaded_file = st.file_uploader("CSV 파일 업로드", type=['csv'])
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        X_w5_up, X_w20_up, labels_up, atypes_up, error = preprocess_and_window(
            df_upload, scaler, FEATURE_NAMES
        )
        if error:
            st.error(error)
            st.warning("누락된 컬럼을 확인하고 다시 업로드해주세요.")
        else:
            st.session_state['X_w5']         = X_w5_up
            st.session_state['X_w20']        = X_w20_up
            st.session_state['labels']       = labels_up
            st.session_state['attack_types'] = atypes_up
            st.session_state['ready']        = True
            st.success(f" 업로드 완료: {len(X_w5_up)}개 샘플 | 컬럼 검증  | 스케일링  | 윈도우 생성 ")

# ── 분석 실행 ─────────────────────────────────────────────────
if st.session_state.get('ready', False):
    st.markdown("---")
    st.markdown("##  2-Tier 침입 탐지 분석")

    X_w5         = st.session_state['X_w5']
    X_w20        = st.session_state['X_w20']
    labels       = st.session_state['labels']
    attack_types = st.session_state['attack_types']

    st.info(f"총 {len(X_w5)}개 샘플 분석 준비 완료")

    if st.button("분석 시작", use_container_width=True, type="primary"):

        status_ph   = st.empty()
        progress_ph = st.empty()

        result_df = predict_2tier_batch(
            X_w5, X_w20,
            mlp_model, mlp_threshold,
            bilstm_model, bilstm_threshold,
            status_ph, progress_ph
        )
        status_ph.empty()

        # ── 결과 요약 ──────────────────────────────────────────
        st.markdown("### 탐지 결과 요약")
        total      = len(result_df)
        normal     = (result_df['최종판정'] == '✅ 정상').sum()
        attack     = (result_df['최종판정'] == '🚨 공격').sum()
        tier2_used = (result_df['2차결과'] != '미검사').sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("총 샘플",   f"{total}개")
        col2.metric("정상 판정", f"{normal}개",     f"{normal/total*100:.1f}%")
        col3.metric("공격 탐지", f"{attack}개",     f"{attack/total*100:.1f}%")
        col4.metric("2차 검사",  f"{tier2_used}개", f"{tier2_used/total*100:.1f}%")

        st.markdown("---")

        # ── 2-Tier 흐름 시각화 ────────────────────────────────
        st.markdown("### 2-Tier 처리 흐름")
        col1, col2 = st.columns(2)

        with col1:
            fig_funnel = go.Figure(go.Funnel(
                y=['전체 트래픽', '1차(MLP) 의심', '2차(Bi-LSTM) 공격'],
                x=[total, tier2_used, attack],
                textinfo="value+percent initial",
                marker_color=['#4C9BE8', '#F5A623', '#E84C4C']
            ))
            fig_funnel.update_layout(title="2-Tier 필터링 흐름", height=350)
            st.plotly_chart(fig_funnel, use_container_width=True)

        with col2:
            fig_pie = px.pie(
                values=[normal, attack],
                names=['정상', '공격'],
                color_discrete_sequence=['#4C9BE8', '#E84C4C'],
                title="최종 탐지 결과"
            )
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── 공격 유형별 탐지율 ────────────────────────────────
        if attack_types is not None and labels is not None:
            st.markdown("### 공격 유형별 탐지율")
            result_df['attack_type'] = attack_types
            result_df['true_label']  = labels

            type_stats = []
            for atype in ['BruteForce', 'DoS', 'DDoS']:
                mask     = result_df['attack_type'] == atype
                true_atk = ((result_df['true_label'] != 'BENIGN') & mask).sum()
                detected = ((result_df['최종판정'] == '🚨 공격') & mask &
                            (result_df['true_label'] != 'BENIGN')).sum()
                recall   = detected / true_atk if true_atk > 0 else 0
                type_stats.append({
                    '공격 유형': atype,
                    '전체 샘플': int(mask.sum()),
                    '실제 공격': int(true_atk),
                    '탐지 성공': int(detected),
                    'Recall'   : f"{recall:.3f}"
                })

            stats_df = pd.DataFrame(type_stats)
            st.dataframe(stats_df, use_container_width=True)

            fig_bar = px.bar(
                stats_df, x='공격 유형', y='Recall',
                color='공격 유형',
                color_discrete_sequence=['#E84C4C', '#F5A623', '#9B59B6'],
                title="공격 유형별 Recall",
                text='Recall'
            )
            fig_bar.update_layout(yaxis_range=[0, 1], height=350)
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── MSE 분포 ──────────────────────────────────────────
        st.markdown("### 재구성 오차 (MSE) 분포")
        col1, col2 = st.columns(2)

        with col1:
            fig_mse1 = px.histogram(
                result_df, x='1차MSE',
                color_discrete_sequence=['#4C9BE8'],
                title="1차 방어선 (MLP) MSE 분포",
                nbins=50
            )
            fig_mse1.add_vline(
                x=mlp_threshold, line_dash="dash",
                line_color="red",
                annotation_text=f"임계값 {mlp_threshold:.6f}"
            )
            st.plotly_chart(fig_mse1, use_container_width=True)

        with col2:
            tier2_df = result_df[result_df['2차MSE'].notna()]
            if len(tier2_df) > 0:
                fig_mse2 = px.histogram(
                    tier2_df, x='2차MSE',
                    color_discrete_sequence=['#F5A623'],
                    title="2차 방어선 (Bi-LSTM) MSE 분포",
                    nbins=50
                )
                fig_mse2.add_vline(
                    x=bilstm_threshold, line_dash="dash",
                    line_color="red",
                    annotation_text=f"임계값 {bilstm_threshold:.6f}"
                )
                st.plotly_chart(fig_mse2, use_container_width=True)
            else:
                st.info("2차 검사 샘플 없음")

        # ── 상세 결과 테이블 ──────────────────────────────────
        st.markdown("### 상세 결과")
        display_df = result_df.copy()
        if labels is not None:
            display_df['실제레이블'] = labels
        if attack_types is not None:
            display_df['공격유형'] = attack_types

        if labels is not None:
            display_df['정오답'] = display_df.apply(
                lambda row: '정답' if (
                    (row['실제레이블'] == 'BENIGN' and row['최종판정'] == '✅ 정상') or
                    (row['실제레이블'] != 'BENIGN' and row['최종판정'] == '🚨 공격')
                ) else '오답', axis=1
            )

        cols = ['샘플', '공격유형', '실제레이블', '1차MSE', '1차결과',
                '2차MSE', '2차결과', '최종판정', '정오답']
        cols = [c for c in cols if c in display_df.columns]
        st.dataframe(display_df[cols], use_container_width=True, height=300)

        # ── 공격 유형별 Recall 메트릭 ─────────────────────────
        if labels is not None and attack_types is not None:
            st.markdown("### 공격 유형별 Recall")
            col1, col2, col3 = st.columns(3)

            for col, atype in zip([col1, col2, col3],
                                  ['BruteForce', 'DoS', 'DDoS']):
                mask     = display_df['공격유형'] == atype
                true_atk = (display_df[mask]['실제레이블'] != 'BENIGN').sum()
                detected = ((display_df[mask]['최종판정'] == '🚨 공격') &
                            (display_df[mask]['실제레이블'] != 'BENIGN')).sum()
                recall   = detected / true_atk if true_atk > 0 else 0

                delta_color = "normal" if recall >= 0.7 else \
                              "off"    if recall >= 0.3 else "inverse"

                col.metric(
                    label=atype,
                    value=f"{recall:.1%}",
                    delta=f"{detected}/{true_atk}개 탐지",
                    delta_color=delta_color
                )