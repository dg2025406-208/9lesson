import math

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --------------------------------------------------
# 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

RAW_COLUMNS = [
    "movieCd",
    "movieNm",
    "openDt",
    "genre",
    "nation",
    "first_scrn",
    "first_show",
    "first_date",
    "peak",
    "first_week_audi",
    "total_audi",
    "days_in_top10",
]

ATTR_LABELS = {
    "스크린 수": "스크린 수",
    "누적 관객": "누적 관객",
    "10위권 일수": "10위권 일수",
    "롱런 지수": "롱런 지수",
}

CLUSTER_SYMBOLS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    missing_columns = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"데이터에 필요한 열이 없습니다: {', '.join(missing_columns)}")

    numeric_cols = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    total_count = len(df)

    # 유형 분류에 사용할 네 속성
    work = df.copy()
    work["스크린 수"] = work["first_scrn"].apply(
        lambda x: math.log10(x) if pd.notna(x) and x > 0 else pd.NA
    )
    work["누적 관객"] = work["total_audi"].apply(
        lambda x: math.log10(x) if pd.notna(x) and x > 0 else pd.NA
    )
    work["10위권 일수"] = work["days_in_top10"]
    work["롱런 지수"] = work.apply(
        lambda row: min(row["total_audi"] / row["first_week_audi"], 20)
        if pd.notna(row["total_audi"])
        and pd.notna(row["first_week_audi"])
        and row["first_week_audi"] > 0
        else pd.NA,
        axis=1,
    )

    feature_cols = list(ATTR_LABELS.keys())
    work[feature_cols] = work[feature_cols].apply(pd.to_numeric, errors="coerce")

    # 네 속성 중 하나라도 없거나 첫 주 관객이 0이면 제외
    valid_mask = work[feature_cols].notna().all(axis=1) & (work["first_week_audi"] > 0)
    work = work.loc[valid_mask].copy()
    clustered_count = len(work)

    return work, total_count, clustered_count


def run_clustering(df, selected_attrs, n_clusters):
    scaler = StandardScaler()
    x_standardized = scaler.fit_transform(df[selected_attrs])

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )
    raw_labels = model.fit_predict(x_standardized)

    result = df.copy()
    result["cluster_raw"] = raw_labels

    # 누적 관객 평균이 큰 묶음부터 ㉮, ㉯, ㉰ ... 순서로 표시
    mean_total = (
        result.groupby("cluster_raw")["total_audi"]
        .mean()
        .sort_values(ascending=False)
    )
    label_map = {
        raw_label: display_label
        for raw_label, display_label in zip(
            mean_total.index, CLUSTER_SYMBOLS[:n_clusters]
        )
    }
    result["묶음"] = result["cluster_raw"].map(label_map)

    return result


# --------------------------------------------------
# 제목
# --------------------------------------------------
st.title("🎬 영화 유형 나누기")
st.caption(
    "영화의 흥행·상영 특성을 바탕으로 비슷한 영화끼리 선택한 수의 유형으로 묶어 봅니다."
)

try:
    df, total_count, clustered_count = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 문제가 생겼습니다: {e}")
    st.stop()

# --------------------------------------------------
# 상단 요약
# --------------------------------------------------
st.write(f"**전체 편수:** {total_count:,}편　|　**묶은 편수:** {clustered_count:,}편")

# --------------------------------------------------
# 묶는 속성 선택
# --------------------------------------------------
st.subheader("1. 영화 유형을 나눌 속성 선택")
selected_attrs = st.multiselect(
    "k-평균에 사용할 속성을 선택하세요. (2개 이상)",
    options=list(ATTR_LABELS.keys()),
    default=list(ATTR_LABELS.keys()),
)

if len(selected_attrs) < 2:
    st.warning("속성을 2개 이상 골라 주세요.")
    st.stop()

# --------------------------------------------------
# 묶음 수 선택
# --------------------------------------------------
st.subheader("2. 묶음 수 선택")
n_clusters = st.slider(
    "몇 개의 유형으로 나눌까요?",
    min_value=2,
    max_value=7,
    value=3,
    step=1,
)

clustered_df = run_clustering(df, selected_attrs, n_clusters)
cluster_order = CLUSTER_SYMBOLS[:n_clusters]

# 선택한 속성을 표준화한 값
scaler = StandardScaler()
x_standardized = scaler.fit_transform(df[selected_attrs])

# --------------------------------------------------
# 1~7개 묶음의 중심에서의 거리 제곱합
# --------------------------------------------------
inertia_rows = []
for k in range(1, 8):
    model_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    model_k.fit(x_standardized)
    inertia_rows.append(
        {
            "묶음 수": k,
            "중심에서의 거리 제곱합": model_k.inertia_,
        }
    )

inertia_df = pd.DataFrame(inertia_rows)
inertia_df["직전 값 대비 감소량"] = (
    inertia_df["중심에서의 거리 제곱합"].shift(1)
    - inertia_df["중심에서의 거리 제곱합"]
)

# --------------------------------------------------
# 선택한 묶음 수의 실루엣 점수
# --------------------------------------------------
silhouette_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
silhouette_labels = silhouette_model.fit_predict(x_standardized)
silhouette = silhouette_score(x_standardized, silhouette_labels)

# --------------------------------------------------
# 2차원 산점도
# --------------------------------------------------
st.subheader("3. 2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_attr = st.selectbox("가로축(X)", selected_attrs, index=0)
with col2:
    y_attr = st.selectbox(
        "세로축(Y)",
        selected_attrs,
        index=1,
    )

fig_2d = px.scatter(
    clustered_df,
    x=x_attr,
    y=y_attr,
    color="묶음",
    category_orders={"묶음": cluster_order},
    hover_name="movieNm",
    hover_data={
        "movieNm": False,
        x_attr: ":.2f",
        y_attr: ":.2f",
        "묶음": True,
    },
    labels={
        x_attr: ATTR_LABELS[x_attr],
        y_attr: ATTR_LABELS[y_attr],
        "묶음": "유형",
    },
    title=f"{ATTR_LABELS[x_attr]} × {ATTR_LABELS[y_attr]}",
)
fig_2d.update_traces(marker={"size": 8})
fig_2d.update_layout(legend_title_text="유형", height=560)
st.plotly_chart(fig_2d, use_container_width=True)

# --------------------------------------------------
# 3차원 산점도
# --------------------------------------------------
st.subheader("4. 3차원 산점도")
if len(selected_attrs) < 3:
    st.info("3차원 산점도를 보려면 유형을 나눌 속성을 3개 이상 선택해 주세요.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        z_x = st.selectbox("X축", selected_attrs, index=0, key="3d_x")
    with c2:
        z_y = st.selectbox("Y축", selected_attrs, index=1, key="3d_y")
    with c3:
        z_z = st.selectbox("Z축", selected_attrs, index=2, key="3d_z")

    fig_3d = px.scatter_3d(
        clustered_df,
        x=z_x,
        y=z_y,
        z=z_z,
        color="묶음",
        category_orders={"묶음": cluster_order},
        hover_name="movieNm",
        hover_data={
            "movieNm": False,
            z_x: ":.2f",
            z_y: ":.2f",
            z_z: ":.2f",
            "묶음": True,
        },
        labels={
            z_x: ATTR_LABELS[z_x],
            z_y: ATTR_LABELS[z_y],
            z_z: ATTR_LABELS[z_z],
            "묶음": "유형",
        },
        title="3차원 영화 유형 분포",
    )
    fig_3d.update_traces(marker={"size": 3})
    fig_3d.update_layout(height=680, legend_title_text="유형")
    st.plotly_chart(fig_3d, use_container_width=True)

# --------------------------------------------------
# 묶음별 평균 표
# --------------------------------------------------
st.subheader("5. 묶음별 특징")

summary = (
    clustered_df.groupby("묶음")
    .agg(
        편수=("movieNm", "count"),
        스크린수_평균=("first_scrn", "mean"),
        누적관객_평균=("total_audi", "mean"),
        십위권일수_평균=("days_in_top10", "mean"),
        롱런지수_평균=("롱런 지수", "mean"),
    )
    .reindex(cluster_order)
    .reset_index()
)

summary_display = summary.rename(
    columns={
        "묶음": "유형",
        "편수": "편수",
        "스크린수_평균": "스크린 수 평균",
        "누적관객_평균": "누적 관객 평균",
        "십위권일수_평균": "10위권 일수 평균",
        "롱런지수_평균": "롱런 지수 평균",
    }
)

summary_display["스크린 수 평균"] = summary_display["스크린 수 평균"].round(1)
summary_display["누적 관객 평균"] = (
    summary_display["누적 관객 평균"].round(0).astype("Int64")
)
summary_display["10위권 일수 평균"] = summary_display["10위권 일수 평균"].round(1)
summary_display["롱런 지수 평균"] = summary_display["롱런 지수 평균"].round(2)

st.dataframe(summary_display, use_container_width=True, hide_index=True)

# --------------------------------------------------
# 묶음별 누적 관객 상위 5편
# --------------------------------------------------
st.subheader("6. 묶음별 누적 관객 상위 5편")

cols = st.columns(n_clusters)
for col, symbol in zip(cols, cluster_order):
    top_movies = (
        clustered_df[clustered_df["묶음"] == symbol]
        .sort_values("total_audi", ascending=False)
        .head(5)
    )

    with col:
        st.markdown(f"### {symbol}")
        for i, row in enumerate(top_movies.itertuples(index=False), start=1):
            st.write(f"{i}. {row.movieNm} ({row.total_audi:,.0f}명)")

# --------------------------------------------------
# 묶음 수에 따른 거리 제곱합
# --------------------------------------------------
st.subheader("7. 묶음 수에 따른 거리 제곱합")
st.caption(
    "선택한 속성을 표준화한 뒤, 각 묶음 안에서 영화가 중심에서 떨어진 거리의 제곱을 모두 더한 값입니다."
)

fig_elbow = px.line(
    inertia_df,
    x="묶음 수",
    y="중심에서의 거리 제곱합",
    markers=True,
    labels={
        "묶음 수": "묶음 수",
        "중심에서의 거리 제곱합": "중심에서의 거리 제곱합",
    },
    title="1~7개 묶음의 중심에서의 거리 제곱합",
)
fig_elbow.add_vline(
    x=n_clusters,
    line_dash="dash",
    annotation_text=f"현재: {n_clusters}개",
    annotation_position="top right",
)
fig_elbow.update_xaxes(dtick=1)
fig_elbow.update_layout(height=500)
st.plotly_chart(fig_elbow, use_container_width=True)

inertia_table = inertia_df.copy()
inertia_table["중심에서의 거리 제곱합"] = inertia_table[
    "중심에서의 거리 제곱합"
].round(2)
inertia_table["직전 값 대비 감소량"] = inertia_table[
    "직전 값 대비 감소량"
].round(2)
inertia_table["직전 값 대비 감소량"] = inertia_table[
    "직전 값 대비 감소량"
].astype("Float64")
inertia_table["직전 값 대비 감소량"] = inertia_table[
    "직전 값 대비 감소량"
].mask(inertia_table["묶음 수"] == 1, pd.NA)

st.dataframe(inertia_table, use_container_width=True, hide_index=True)

# --------------------------------------------------
# 실루엣 점수
# --------------------------------------------------
st.subheader("8. 현재 묶음 수의 실루엣 점수")
st.write(f"**현재 {n_clusters}개 묶음의 실루엣 점수:** {silhouette:.3f}")
st.caption(
    "실루엣 점수는 -1에서 1 사이이며, 1에 가까울수록 묶음이 뚜렷하게 나뉜다는 뜻입니다."
)
