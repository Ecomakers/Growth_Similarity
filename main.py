import streamlit as st
import pandas as pd
import numpy as np
import os
import re
from tslearn.metrics import dtw_path
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 설정
DATA_DIR = "data"

# 컬럼 정리 함수
def clean_column(col):
    return re.sub(r"\(.*?\)", "", col[0]).strip()

# DTW 계산 함수
def calculate_dtw(njc_df, sample_df, hangmoks):
    dtw_total = pd.DataFrame()

    for year in njc_df['crps_year'].unique():
        temp = njc_df[njc_df['crps_year'] == year]

        njc_mean = temp.groupby('주차')[hangmoks].mean().reset_index()
        njc_min = temp.groupby('주차')[hangmoks].min().reset_index()
        njc_max = temp.groupby('주차')[hangmoks].max().reset_index()

        njc_mean.fillna(method='bfill', inplace=True)
        njc_min.fillna(method='bfill', inplace=True)
        njc_max.fillna(method='bfill', inplace=True)

        dtw_distance = []
        dtw_max_distance = []

        for hangmok in hangmoks:
            try:
                A = njc_mean[(njc_mean['주차'] >= sample_df['주차'].min()) & (njc_mean['주차'] <= sample_df['주차'].max())][hangmok]
                A_max = njc_max[(njc_max['주차'] >= sample_df['주차'].min()) & (njc_max['주차'] <= sample_df['주차'].max())][hangmok]
                A_min = njc_min[(njc_min['주차'] >= sample_df['주차'].min()) & (njc_min['주차'] <= sample_df['주차'].max())][hangmok]
                B = sample_df.groupby('주차')[hangmok].mean()

                _, distance = dtw_path(A.values, B.values)
                _, max_distance = dtw_path(A_max.values, A_min.values)

                dtw_distance.append(distance)
                dtw_max_distance.append(max_distance)
            except Exception as e:
                dtw_distance.append(np.nan)
                dtw_max_distance.append(np.nan)

        dtw_df = pd.DataFrame({
            '항목': hangmoks,
            '거리': dtw_distance,
            '최대거리': dtw_max_distance,
            '연도': year
        })

        dtw_total = pd.concat([dtw_total, dtw_df])

    dtw_calc = dtw_total.groupby('항목')[['거리', '최대거리']].mean()
    dtw_calc['생육유사도(%)'] = round((1 - dtw_calc['거리'] / dtw_calc['최대거리']) * 100, 2)
    return dtw_calc.reset_index()


# Streamlit 앱
st.title("🌱 생육 유사도 비교")

st.sidebar.header("📤 샘플 Excel 업로드")
uploaded_file = st.sidebar.file_uploader("엑셀 파일을 선택하세요", type=["xlsx"])

# 업로드 안내 메시지
if not uploaded_file:
    st.info('''👈 왼쪽 사이드바에서 샘플 Excel 파일을 업로드해주세요.  \n  \n
    ⚠️딸기, 완숙토마토, 방울토마토, 오이, 파프리카 품목만 생육유사도 측정이 가능합니다.''')

# 2. 샘플 Excel 업로드
if uploaded_file:
    progress = st.progress(0, text="📊 파일 처리 중입니다...")

    # 1단계: 샘플 데이터 읽기
    sample_df = pd.read_excel(uploaded_file, header=[0, 1, 2])
    sample_df = sample_df[sample_df.columns[:-1]]  # 마지막 열 제거
    sample_df.columns = [clean_column(col) for col in sample_df.columns]
    sample_df = sample_df.rename(columns={'줄기직경': '줄기굵기'})

    progress.progress(10, text="✅ 컬럼 정리 완료")

    if '주차' not in sample_df.columns:
        st.error("샘플 데이터에 '주차' 열이 있어야 합니다.")
        progress.empty()
    else:
        # 2단계: 작물 자동 인식
        crop_mapping = {
            '딸기': ('strawberry.csv', ['초장','엽장','엽폭','엽수','엽병장','관부직경']),
            '완숙토마토': ('tomatoes.csv', ['생장길이','줄기굵기','엽장','엽폭','엽수','화방높이']),
            '방울토마토': ('cherry_tomatoes.csv', ['생장길이','줄기굵기','엽장','엽폭','엽수','화방높이']),
            '오이': ('cucumber.csv', ['초장','마디수','줄기굵기','엽장','엽폭','엽수']),
            '파프리카': ('paprika.csv', ['생장길이','줄기굵기','엽장','엽폭','엽수','개화마디','착과마디']),
        }

        # 작물 일치율 계산 방식으로 개선
        matched_crops = []
        for crop_name, (_, crop_hangmoks) in crop_mapping.items():
            matched = [h for h in crop_hangmoks if h in sample_df.columns]
            match_ratio = len(matched) / len(crop_hangmoks)
            matched_crops.append({
                "name": crop_name,
                "matched_count": len(matched),
                "total_required": len(crop_hangmoks),
                "match_ratio": match_ratio
            })

        # 1순위: 일치 개수 > 2순위: 일치율 기준 정렬
        matched_crops.sort(key=lambda x: (x["matched_count"], x["match_ratio"]), reverse=True)
        top_match_count = matched_crops[0]['matched_count']

        # 최소 일치 개수 조건 확인 (예: 3개 이상)
        if top_match_count < 3:
            st.error("업로드된 파일이 어떤 작물에도 해당하지 않습니다.")
            progress.empty()
        else:
            top_candidates = [c for c in matched_crops if c['matched_count'] == top_match_count]

            if len(top_candidates) == 1:
                crop = top_candidates[0]['name']
                st.success(f"자동 인식된 작물: {crop}")
            else:
                crop_names = [c['name'] for c in top_candidates]
                crop = st.selectbox("🔍 여러 작물이 유사하게 감지되었습니다. 작물을 선택해주세요:", crop_names)
                st.success(f"선택된 작물: {crop}")


            filename, hangmoks = crop_mapping[crop]
            njc_df = pd.read_csv(f"{DATA_DIR}/{filename}", encoding='cp949')

            progress.progress(40, text="📂 농진청 데이터 로딩 중...")

            sample_data = sample_df[['주차'] + [col for col in sample_df.columns if col in hangmoks]]

            progress.progress(60, text="🔄 DTW 유사도 계산 중...")

            dtw_result = calculate_dtw(njc_df, sample_data, hangmoks)

            progress.progress(80, text="📈 시각화 준비 중...")

            # --- 항목별 주차별 비교 시각화 ---
            st.subheader("📉 주차별 생육 비교 (사용자 데이터 vs 농진청 데이터)")

            tabs = st.tabs(hangmoks)

            for i, hangmok in enumerate(hangmoks):
                with tabs[i]:
                    # 샘플 데이터 평균
                    sample_grouped = sample_data.groupby('주차')[hangmok].mean().reset_index()

                    # NJC 데이터 평균 (최근 연도 기준)
                    recent_year = njc_df['crps_year'].max()
                    njc_filtered = njc_df[njc_df['crps_year'] == recent_year]
                    njc_grouped = njc_filtered.groupby('주차')[hangmok].mean().reset_index()

                    # plotly 그래프 생성
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=njc_grouped['주차'],
                        y=njc_grouped[hangmok],
                        mode='lines+markers',
                        name='농진청 평균',
                        marker=dict(color='green')
                    ))

                    fig.add_trace(go.Scatter(
                        x=sample_grouped['주차'],
                        y=sample_grouped[hangmok],
                        mode='lines+markers',
                        name='사용자 평균',
                        marker=dict(color='blue')
                    ))

                    fig.update_layout(
                        title=f"",
                        xaxis_title="주차",
                        yaxis_title=hangmok,
                        legend=dict(x=0.01, y=0.99),
                        height=400,
                        hovermode="x unified",       # 마우스 오버 시 x축 기준 통합 툴팁
                        title_font_size=1,          # 타이틀 폰트 크기
                        title_x=0.01,                 # 타이틀 가운데 정렬 (x축 기준)
                        title_y=0.9,                  # 타이틀 y 위치 조정
                        margin=dict(t=20)
                    )

                    st.plotly_chart(fig, use_container_width=True)
            progress.progress(90, text="주차별 생육비교 시각화 완료..")
            # st.subheader("📊 생육 유사도 (DTW 기반)")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 🔍 유사도 테이블")
                st.dataframe(dtw_result.reset_index(drop=True))

            progress.progress(95, text="유사도 테이블 완료..")



            with col2:
                st.markdown("#### 📈 유사도 시각화 (Radar Chart)")

                # 데이터 준비
                categories = dtw_result['항목'].tolist()
                values = dtw_result['생육유사도(%)'].tolist()

                # 레이더 차트 닫기 위해 첫 항목 반복
                categories += [categories[0]]
                values += [values[0]]

                # 레이더 차트 생성
                fig = go.Figure()

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='생육유사도(%)',
                    line_color='skyblue',
                    hovertemplate='<b>%{theta}</b><br>유사도: %{r:.1f}%<extra></extra>'
                ))
                # 레이아웃 조정: 마진 최소화 + 크기 조절
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100],
                            tickfont=dict(size=10)  # 폰트도 조금 작게
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=10)  # 각 항목 이름도 작게
                        )
                    ),
                    margin=dict(t=20, b=20, l=40, r=40),  # 여백 최소화
                    height=350,  # 높이 줄이기
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)
        progress.progress(100, text="레이더 차트 완료!")
        progress.empty()
