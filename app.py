"""
Web App Streamlit: Dự đoán hỏng hóc máy phục vụ bảo trì dự đoán
trên dữ liệu cảm biến mất cân bằng (Bộ dữ liệu AI4I 2020)
Nhóm 16 - Trường ĐH Công nghệ Thông tin và Truyền thông (ICTU)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from train_eval import (
    ARTIFACTS_PATH,
    CHARTS_DIR,
    INPUT_COLS_RAW,
    NUMERIC_COLS_CLEAN,
    NUMERIC_COLS_RAW,
    RESULTS_DIR,
    apply_scaler,
    compute_metrics,
    encode_and_clean_features,
    normalize_input_columns,
    read_csv_smart,
    run_pipeline,
)

# ==============================================================================
# CẤU HÌNH TRANG & CSS HIỆN ĐẠI
# ==============================================================================
st.set_page_config(
    page_title="Dự đoán Hỏng hóc Máy - AI4I 2020 | Nhóm 16 ICTU",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* Nền & khoảng cách tổng thể */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }

    /* Banner tiêu đề chính */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 58%, #0284c7 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        color: #ffffff;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.16);
        margin-bottom: 1.25rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.16);
        color: #e0f2fe;
        padding: 0.22rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 1.55rem;
        font-weight: 800;
        margin: 0.15rem 0 0.35rem 0;
        line-height: 1.3;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin: 0;
    }

    /* Thẻ KPI bo góc, đổ bóng nhẹ */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        height: 100%;
    }
    .kpi-label {
        font-size: 0.80rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.80rem;
        color: #475569;
        margin-top: 0.3rem;
    }

    /* Hộp trạng thái dự đoán */
    .status-card-ok {
        background: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 1.15rem 1.35rem;
        color: #065f46;
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.12);
        margin-bottom: 1rem;
    }
    .status-card-fail {
        background: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 1.15rem 1.35rem;
        color: #991b1b;
        box-shadow: 0 6px 16px rgba(239, 68, 68, 0.14);
        margin-bottom: 1rem;
    }
    .status-title {
        font-size: 1.15rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }
    .status-desc {
        font-size: 0.92rem;
        margin: 0;
        line-height: 1.5;
    }

    /* Thẻ thông tin mẫu test */
    .sample-meta-box {
        background: #f8fafc;
        border-left: 4px solid #0284c7;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        color: #0f172a;
        margin: 0.5rem 0 1rem 0;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }

    /* Thẻ thông số sidebar */
    .sidebar-card {
        background: rgba(15, 23, 42, 0.04);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.85rem;
    }

    /* Tabs đẹp hơn */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# NẠP ARTIFACTS & DỮ LIỆU KẾT QUẢ
# ==============================================================================
def ensure_artifacts_ready() -> None:
    """Nếu thiếu model_artifacts.pkl hoặc file kết quả, tự động gọi train_eval.run_pipeline()."""
    required_files = [
        ARTIFACTS_PATH,
        RESULTS_DIR / "1_thong_ke_mo_ta.csv",
        RESULTS_DIR / "2_phan_chia_du_lieu.csv",
        RESULTS_DIR / "3_ket_qua_validation.csv",
        RESULTS_DIR / "4_danh_gia_qua_khop.csv",
        RESULTS_DIR / "5_ket_qua_test.csv",
        CHARTS_DIR / "label_distribution.png",
        CHARTS_DIR / "correlation_matrix.png",
        CHARTS_DIR / "pr_curve_validation.png",
        CHARTS_DIR / "threshold_tuning.png",
        CHARTS_DIR / "confusion_matrix_test.png",
        CHARTS_DIR / "feature_importance.png",
    ]
    if any(not path.exists() for path in required_files):
        with st.spinner("Chưa phát hiện đầy đủ model_artifacts.pkl / kết quả thực nghiệm. Đang tự động huấn luyện từ train_eval.py..."):
            run_pipeline()


@st.cache_resource
def load_model_artifacts() -> Dict[str, Any]:
    return joblib.load(ARTIFACTS_PATH)


@st.cache_data
def load_csv_result(filename: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / filename)


ensure_artifacts_ready()
artifacts = load_model_artifacts()

models_dict: Dict[str, Dict[str, Any]] = artifacts["models"]
best_model_name: str = artifacts["best_model_name"]
best_threshold: float = float(artifacts["best_threshold"])
scaler = artifacts["scaler"]
df_test: pd.DataFrame = artifacts["df_test"]
test_probs_best: np.ndarray = artifacts["test_probs"]
quality_summary: Dict[str, Any] = artifacts["quality_summary"]

df_desc = load_csv_result("1_thong_ke_mo_ta.csv")
df_split = load_csv_result("2_phan_chia_du_lieu.csv")
df_val = load_csv_result("3_ket_qua_validation.csv")
df_overfit = load_csv_result("4_danh_gia_qua_khop.csv")
df_test_summary = load_csv_result("5_ket_qua_test.csv")


# ==============================================================================
# HÀM HỖ TRỢ DỰ ĐOÁN & TRỰC QUAN HÓA
# ==============================================================================
def predict_dataframe(
    df_features: pd.DataFrame, model_name: str, threshold: float
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Chạy đúng pipeline Mã hóa -> (Chuẩn hóa nếu LR) -> Dự đoán xác suất & nhãn."""
    X_encoded = encode_and_clean_features(df_features)
    X_scaled = apply_scaler(X_encoded, scaler, fit=False)

    model_info = models_dict[model_name]
    clf = model_info["model"]
    X_model_in = X_scaled if model_info["uses_scaled"] else X_encoded

    probs = clf.predict_proba(X_model_in)[:, 1]
    preds = (probs >= threshold).astype(int)
    return probs, preds, X_encoded, X_scaled


def build_gauge_chart(prob: float, threshold: float) -> go.Figure:
    """Biểu đồ đồng hồ đo Plotly Gauge Chart thể hiện P(y=1|x) so với ngưỡng t*."""
    prob_pct = prob * 100.0
    thr_pct = threshold * 100.0

    bar_color = "#dc2626" if prob >= threshold else "#10b981"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=prob_pct,
            number={"suffix": "%", "valueformat": ".2f", "font": {"size": 34, "color": "#0f172a"}},
            delta={
                "reference": thr_pct,
                "valueformat": ".2f",
                "suffix": "%",
                "increasing": {"color": "#dc2626"},
                "decreasing": {"color": "#10b981"},
            },
            title={"text": f"Xác suất Hỏng hóc P(y=1|x)<br><span style='font-size:13px;color:#64748b'>Ngưỡng quyết định t = {threshold:.2f} ({thr_pct:.0f}%)</span>", "font": {"size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#475569"},
                "bar": {"color": bar_color, "thickness": 0.30},
                "bgcolor": "#ffffff",
                "borderwidth": 1.5,
                "bordercolor": "#cbd5e1",
                "steps": [
                    {"range": [0, max(0, thr_pct * 0.75)], "color": "#dcfce7"},
                    {"range": [max(0, thr_pct * 0.75), thr_pct], "color": "#fef9c3"},
                    {"range": [thr_pct, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#b91c1c", "width": 4},
                    "thickness": 0.85,
                    "value": thr_pct,
                },
            },
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=25, r=25, t=55, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def analyze_engineering_signals(
    prod_type: str, air_t: float, proc_t: float, rpm: float, torque: float, wear: float
) -> List[Dict[str, str]]:
    """Phân tích kỹ thuật nhanh dựa trên đặc trưng vật lý của AI4I 2020."""
    delta_t = proc_t - air_t
    omega = 2.0 * math.pi * rpm / 60.0
    power_w = torque * omega
    osf_load = torque * wear

    osf_limits = {"L": 11000.0, "M": 12000.0, "H": 13000.0}
    osf_limit = osf_limits.get(prod_type, 11000.0)

    signals: List[Dict[str, str]] = []

    # 1. Tản nhiệt (HDF)
    if delta_t < 8.6 and rpm < 1380:
        signals.append({
            "icon": "🔥",
            "title": "Nguy cơ Lỗi Tản nhiệt (HDF)",
            "detail": f"Chênh lệch nhiệt ΔT = {delta_t:.1f} K (< 8.6 K) kết hợp tốc độ quay thấp {rpm:.0f} rpm (< 1.380 rpm).",
            "level": "danger",
        })
    elif delta_t < 9.2:
        signals.append({
            "icon": "🌡️",
            "title": "Chênh lệch nhiệt độ thấp",
            "detail": f"ΔT = {delta_t:.1f} K tiệm cận ngưỡng nhạy cảm tản nhiệt (8.6 K).",
            "level": "warning",
        })
    else:
        signals.append({
            "icon": "❄️",
            "title": "Tản nhiệt ổn định",
            "detail": f"Chênh lệch nhiệt ΔT = {delta_t:.1f} K nằm trong vùng an toàn.",
            "level": "ok",
        })

    # 2. Công suất (PWF)
    if power_w < 3500.0 or power_w > 9000.0:
        signals.append({
            "icon": "⚡",
            "title": "Cảnh báo Công suất Bất thường (PWF)",
            "detail": f"Công suất cơ học P = {power_w:,.0f} W nằm ngoài vùng vận hành chuẩn [3.500 – 9.000 W].",
            "level": "danger",
        })
    else:
        signals.append({
            "icon": "⚡",
            "title": "Công suất cơ học hợp lý",
            "detail": f"P = Torque × ω = {power_w:,.0f} W (trong vùng chuẩn 3.500 – 9.000 W).",
            "level": "ok",
        })

    # 3. Quá tải ứng suất (OSF)
    if osf_load > osf_limit:
        signals.append({
            "icon": "🛠️",
            "title": "Cảnh báo Quá tải Ứng suất (OSF)",
            "detail": f"Tích Torque × Tool wear = {osf_load:,.0f} Nm·min vượt giới hạn {osf_limit:,.0f} đối với cấp {prod_type}.",
            "level": "danger",
        })
    elif osf_load > osf_limit * 0.85:
        signals.append({
            "icon": "⚠️",
            "title": "Ứng suất cơ học cao",
            "detail": f"Torque × Tool wear = {osf_load:,.0f} Nm·min đạt {osf_load/osf_limit*100:.0f}% ngưỡng giới hạn ({osf_limit:,.0f}).",
            "level": "warning",
        })
    else:
        signals.append({
            "icon": "🔩",
            "title": "Ứng suất cơ học bình thường",
            "detail": f"Torque × Tool wear = {osf_load:,.0f} Nm·min (< {osf_limit:,.0f}).",
            "level": "ok",
        })

    # 4. Hao mòn dụng cụ (TWF)
    if wear >= 200:
        signals.append({
            "icon": "⏳",
            "title": "Hao mòn dụng cụ cao (TWF)",
            "detail": f"Tool wear = {wear:.0f} phút (≥ 200 phút) — nên lên kế hoạch kiểm tra/thay dao.",
            "level": "warning",
        })
    else:
        signals.append({
            "icon": "✅",
            "title": "Thời gian sử dụng dụng cụ ổn định",
            "detail": f"Tool wear = {wear:.0f} phút (< 200 phút).",
            "level": "ok",
        })

    return signals


def get_id_col_name(df_frame: pd.DataFrame) -> str:
    return "UDI" if "UDI" in df_frame.columns else ("UID" if "UID" in df_frame.columns else "Index")


# ==============================================================================
# SIDEBAR THÔNG TIN ĐỀ TÀI & MÔ HÌNH
# ==============================================================================
with st.sidebar:
    st.markdown("### ⚙️ BẢO TRÌ DỰ ĐOÁN AI4I 2020")
    st.markdown(
        """
        <div class="sidebar-card">
            <strong>Đơn vị:</strong> Trường ĐH CNTT & Truyền thông (ICTU)<br>
            <strong>Nhóm thực hiện:</strong> Nhóm 16<br>
            <span style="font-size:0.86rem;">• Vi Hải Nam<br>• Lý Nhật Quang<br>• Nguyễn Anh Hào</span><br>
            <strong>GVHD:</strong> TS. Ngô Hữu Huy
        </div>
        """,
        unsafe_allow_html=True,
    )

    best_val_row = df_val.loc[df_val["Mô hình"] == best_model_name].iloc[0]
    best_test_row = df_test_summary.iloc[1]

    st.markdown("#### 🏆 Mô hình tối ưu đã chọn")
    st.success(f"**{best_model_name}**")
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Val AP", f"{best_val_row['AP']:.4f}")
    col_s2.metric("Ngưỡng t*", f"{best_threshold:.2f}")

    col_s3, col_s4 = st.columns(2)
    col_s3.metric("Test F1 (t*)", f"{best_test_row['F1-score']:.4f}")
    col_s4.metric("Test Recall", f"{best_test_row['Recall']:.4f}")

    st.markdown("#### 📦 Thông tin bộ dữ liệu")
    st.markdown(
        f"""
        - **Nguồn:** UCI AI4I 2020 (`{quality_summary['n_rows']:,}` mẫu × `{quality_summary['n_cols']}` cột)
        - **Lớp 0 (Không hỏng):** `{quality_summary['n_neg']:,}` (`96.61%`)
        - **Lớp 1 (Có hỏng):** `{quality_summary['n_pos']:,}` (`3.39%`)
        - **Chia tập (Stratified):** `70%` Train (`7.000`) / `15%` Val (`1.500`) / `15%` Test (`1.500`)
        - **Đặc trưng đầu vào:** `6` thuộc tính gốc → `7` đặc trưng sau One-Hot
        """
    )

# ==============================================================================
# BANNER TIÊU ĐỀ CHÍNH
# ==============================================================================
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge">HỌC PHẦN HỌC MÁY • NHÓM 16 – ICTU • PREDICTIVE MAINTENANCE</div>
        <div class="hero-title">Dự đoán Hỏng hóc Máy phục vụ Bảo trì Dự đoán trên Dữ liệu Cảm biến Mất cân bằng</div>
        <p class="hero-subtitle">
            Quy trình chuẩn học máy có giám sát trên bộ dữ liệu AI4I 2020: chống rò rỉ nhãn (Label Leakage),
            xử lý mất cân bằng trọng số lớp, tối ưu ngưỡng phân loại theo F1-score trên Validation và kiểm thử độc lập trên Test.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 4 TAB CHÍNH
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Dự đoán & Chẩn đoán Bảo trì (Thời gian thực)",
    "📂 Dự đoán Hàng loạt qua File CSV",
    "📊 Quy trình Tiền xử lý & Khám phá Dữ liệu (Chương 2)",
    "🏆 Kết quả Thực nghiệm Train - Validation - Test (Chương 3)",
])

# ------------------------------------------------------------------------------
# TAB 1: DỰ ĐOÁN & CHẨN ĐOÁN BẢO TRÌ (THỜI GIAN THỰC)
# ------------------------------------------------------------------------------
with tab1:
    # Khởi tạo session_state cho 6 thông số đầu vào
    default_sample = df_test.iloc[0]
    state_defaults = {
        "inp_type": str(default_sample["Type"]),
        "inp_air": float(default_sample["Air temperature [K]"]),
        "inp_proc": float(default_sample["Process temperature [K]"]),
        "inp_rpm": int(default_sample["Rotational speed [rpm]"]),
        "inp_torque": float(default_sample["Torque [Nm]"]),
        "inp_wear": int(default_sample["Tool wear [min]"]),
        "selected_test_idx": None,
    }
    for k, v in state_defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    st.subheader("1. Chọn nhanh mẫu thực tế từ tập Kiểm thử (Test set - 1.500 mẫu) hoặc tự nhập")

    id_col = get_id_col_name(df_test)
    mode_col, sample_col = st.columns([1.2, 2.2])
    with mode_col:
        sample_mode = st.radio(
            "Nguồn thông số đầu vào:",
            [
                "✏️ Nhập thủ công",
                "🟢 Mẫu Bình thường thực tế (Nhãn 0)",
                "🔴 Mẫu Hỏng hóc thực tế (Nhãn 1)",
            ],
            index=0,
            key="tab1_sample_mode",
        )

    selected_row = None
    with sample_col:
        if sample_mode != "✏️ Nhập thủ công":
            target_val = 0 if "Nhãn 0" in sample_mode else 1
            subset_df = df_test[df_test["Machine failure"] == target_val].copy()

            options = subset_df.index.tolist()

            def fmt_option(idx: int) -> str:
                r = df_test.loc[idx]
                return (
                    f"{id_col} {r[id_col]} | {r['Product ID']} | Type {r['Type']} | "
                    f"Air={r['Air temperature [K]']}K, Proc={r['Process temperature [K]']}K, "
                    f"RPM={r['Rotational speed [rpm]']}, Torque={r['Torque [Nm]']}Nm, Wear={r['Tool wear [min]']}m"
                )

            def on_sample_change() -> None:
                idx = st.session_state["test_sample_picker"]
                r = df_test.loc[idx]
                st.session_state["inp_type"] = str(r["Type"])
                st.session_state["inp_air"] = float(r["Air temperature [K]"])
                st.session_state["inp_proc"] = float(r["Process temperature [K]"])
                st.session_state["inp_rpm"] = int(r["Rotational speed [rpm]"])
                st.session_state["inp_torque"] = float(r["Torque [Nm]"])
                st.session_state["inp_wear"] = int(r["Tool wear [min]"])
                st.session_state["selected_test_idx"] = int(idx)

            # Tự đồng bộ lần đầu khi chuyển sang chế độ chọn mẫu Test
            if (
                "test_sample_picker" not in st.session_state
                or st.session_state["test_sample_picker"] not in options
            ):
                st.session_state["test_sample_picker"] = options[0]
                on_sample_change()

            picked_idx = st.selectbox(
                f"Chọn một mẫu từ tập Test ({len(options):,} mẫu khả dụng):",
                options=options,
                format_func=fmt_option,
                key="test_sample_picker",
                on_change=on_sample_change,
            )
            # Đảm bảo cập nhật nếu người dùng vừa đổi từ "Nhập thủ công" sang mẫu thực tế
            if st.session_state.get("selected_test_idx") != picked_idx:
                on_sample_change()

            selected_row = df_test.loc[picked_idx]
            sub_failures = [c for c in ["TWF", "HDF", "PWF", "OSF", "RNF"] if c in selected_row and int(selected_row[c]) == 1]
            sub_fail_str = ", ".join(sub_failures) if sub_failures else "Không ghi nhận lỗi thành phần"
            actual_badge = "🔴 CÓ HỎNG (Nhãn = 1)" if int(selected_row["Machine failure"]) == 1 else "🟢 BÌNH THƯỜNG (Nhãn = 0)"

            st.markdown(
                f"""
                <div class="sample-meta-box">
                    <strong>Mẫu kiểm thử đang nạp:</strong> <code>{id_col} = {selected_row[id_col]}</code> &nbsp;|&nbsp;
                    <strong>Mã sản phẩm:</strong> <code>{selected_row['Product ID']}</code> &nbsp;|&nbsp;
                    <strong>Nhãn thực tế:</strong> {actual_badge}<br>
                    <span style="font-size:0.86rem; color:#475569;">
                        <strong>Ghi chú lỗi thành phần gốc (chỉ để đối chiếu, không đưa vào mô hình):</strong> {sub_fail_str}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.session_state["selected_test_idx"] = None
            st.info("Đang ở chế độ **Nhập thủ công**. Bạn có thể kéo các thanh trượt bên dưới để mô phỏng trạng thái cảm biến bất kỳ.")

    st.divider()

    left_col, right_col = st.columns([1.05, 1.35], gap="large")

    with left_col:
        st.subheader("2. Thông số vận hành & Cấu hình mô hình")
        model_names = list(models_dict.keys())
        default_m_idx = model_names.index(best_model_name)
        chosen_model = st.selectbox(
            "Chọn mô hình dự đoán:",
            options=model_names,
            index=default_m_idx,
            help="Mặc định sử dụng mô hình đạt Average Precision (AP) cao nhất trên tập Validation.",
        )

        default_thr_for_ui = best_threshold if chosen_model == best_model_name else 0.50
        thr_key = f"thr_{chosen_model}"
        chosen_threshold = st.slider(
            "Ngưỡng phân loại (t):",
            min_value=0.05,
            max_value=0.95,
            value=float(default_thr_for_ui),
            step=0.01,
            key=thr_key,
            help=f"Ngưỡng tối ưu tìm được trên Validation cho {best_model_name} là t* = {best_threshold:.2f}.",
        )

        c1, c2 = st.columns(2)
        with c1:
            inp_type = st.selectbox(
                "Loại sản phẩm (Type):",
                options=["L", "M", "H"],
                key="inp_type",
                help="L: Chất lượng thấp (Low), M: Trung bình (Medium), H: Cao (High)",
            )
            inp_air = st.slider(
                "Nhiệt độ không khí - Air temp [K]:",
                min_value=290.0,
                max_value=310.0,
                step=0.1,
                key="inp_air",
            )
            inp_rpm = st.slider(
                "Tốc độ quay - Rotational speed [rpm]:",
                min_value=1000,
                max_value=3000,
                step=5,
                key="inp_rpm",
            )
        with c2:
            inp_wear = st.slider(
                "Thời gian dùng dao - Tool wear [min]:",
                min_value=0,
                max_value=300,
                step=1,
                key="inp_wear",
            )
            inp_proc = st.slider(
                "Nhiệt độ quá trình - Process temp [K]:",
                min_value=300.0,
                max_value=320.0,
                step=0.1,
                key="inp_proc",
            )
            inp_torque = st.slider(
                "Mô-men xoắn - Torque [Nm]:",
                min_value=0.0,
                max_value=90.0,
                step=0.5,
                key="inp_torque",
            )

    # Thực hiện dự đoán thời gian thực
    df_single = pd.DataFrame([
        {
            "Type": inp_type,
            "Air temperature [K]": float(inp_air),
            "Process temperature [K]": float(inp_proc),
            "Rotational speed [rpm]": int(inp_rpm),
            "Torque [Nm]": float(inp_torque),
            "Tool wear [min]": int(inp_wear),
        }
    ])
    probs_1, preds_1, X_enc_1, X_scl_1 = predict_dataframe(df_single, chosen_model, chosen_threshold)
    prob_val = float(probs_1[0])
    pred_val = int(preds_1[0])

    with right_col:
        st.subheader("3. Kết quả Chẩn đoán & Phân tích Kỹ thuật")
        if pred_val == 1:
            st.markdown(
                f"""
                <div class="status-card-fail">
                    <div class="status-title">🚨 CẢNH BÁO: CÓ NGUY CƠ HỎNG HÓC (DỰ ĐOÁN = 1)</div>
                    <p class="status-desc">
                        Mô hình <strong>{chosen_model}</strong> ước lượng xác suất hỏng hóc đạt
                        <strong>{prob_val*100:.2f}%</strong> (vượt ngưỡng quyết định <strong>{chosen_threshold*100:.0f}%</strong>).
                        Đề xuất lên lịch kiểm tra và bảo trì thiết bị sớm.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="status-card-ok">
                    <div class="status-title">✅ BÌNH THƯỜNG: KHÔNG GHI NHẬN HỎNG HÓC (DỰ ĐOÁN = 0)</div>
                    <p class="status-desc">
                        Mô hình <strong>{chosen_model}</strong> ước lượng xác suất hỏng hóc là
                        <strong>{prob_val*100:.2f}%</strong> (dưới ngưỡng quyết định <strong>{chosen_threshold*100:.0f}%</strong>).
                        Thiết bị đang hoạt động trong vùng an toàn.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        g_col, sig_col = st.columns([1.05, 1.0])
        with g_col:
            st.plotly_chart(build_gauge_chart(prob_val, chosen_threshold), width="stretch")

        with sig_col:
            st.markdown("**Phân tích kỹ thuật nhanh (Chỉ số vật lý AI4I 2020):**")
            eng_signals = analyze_engineering_signals(
                inp_type, float(inp_air), float(inp_proc), float(inp_rpm), float(inp_torque), float(inp_wear)
            )
            for sig in eng_signals:
                if sig["level"] == "danger":
                    st.error(f"{sig['icon']} **{sig['title']}**: {sig['detail']}")
                elif sig["level"] == "warning":
                    st.warning(f"{sig['icon']} **{sig['title']}**: {sig['detail']}")
                else:
                    st.success(f"{sig['icon']} **{sig['title']}**: {sig['detail']}")

        # Đối chiếu trực tiếp với nhãn thực tế nếu đang chọn mẫu từ Test
        if selected_row is not None:
            actual_y = int(selected_row["Machine failure"])
            if actual_y == 1 and pred_val == 1:
                verdict = "TP — Phát hiện đúng hỏng hóc"
            elif actual_y == 0 and pred_val == 0:
                verdict = "TN — Dự đoán đúng bình thường"
            elif actual_y == 0 and pred_val == 1:
                verdict = "FP — Cảnh báo nhầm (Dương tính giả)"
            else:
                verdict = "FN — Bỏ sót hỏng hóc (Âm tính giả)"

            compare_df = pd.DataFrame([
                {
                    "Mã UDI": selected_row[id_col],
                    "Product ID": selected_row["Product ID"],
                    "Nhãn thực tế": f"{actual_y} ({'Có hỏng' if actual_y == 1 else 'Không hỏng'})",
                    "Nhãn dự đoán": f"{pred_val} ({'Có hỏng' if pred_val == 1 else 'Không hỏng'})",
                    "Xác suất P(y=1|x)": f"{prob_val*100:.2f}%",
                    "Ngưỡng": f"{chosen_threshold:.2f}",
                    "Kết luận đối chiếu": verdict,
                }
            ])
            st.markdown("**Đối chiếu với Nhãn thực tế (Tập Test):**")
            st.dataframe(compare_df, width="stretch", hide_index=True)

        # Hiển thị vector 7 đặc trưng sau Mã hóa & Chuẩn hóa
        with st.expander("🔎 Xem vector 7 đặc trưng sau Mã hóa One-Hot & Chuẩn hóa StandardScaler", expanded=True):
            vec_df = pd.DataFrame({
                "Đặc trưng đầu vào (7 biến)": X_enc_1.columns,
                "Sau Mã hóa (Dùng cho RF / XGBoost / LightGBM)": [round(float(v), 4) for v in X_enc_1.iloc[0].values],
                "Sau StandardScaler (Dùng cho Logistic Regression)": [round(float(v), 4) for v in X_scl_1.iloc[0].values],
            })
            st.dataframe(vec_df, width="stretch", hide_index=True)


# ------------------------------------------------------------------------------
# TAB 2: DỰ ĐOÁN HÀNG LOẠT QUA FILE CSV
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Dự đoán Hàng loạt qua File CSV")
    st.markdown(
        "Tải lên file `.csv` chứa 6 thuộc tính đầu vào (`Type`, `Air temperature [K]`, `Process temperature [K]`, "
        "`Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`) hoặc bấm nút dùng thử **50 mẫu trích từ tập Test**."
    )

    b_col1, b_col2, b_col3 = st.columns([1.2, 1.2, 1.4])
    with b_col1:
        batch_model = st.selectbox(
            "Mô hình áp dụng cho dự đoán hàng loạt:",
            options=list(models_dict.keys()),
            index=list(models_dict.keys()).index(best_model_name),
            key="batch_model_select",
        )
    with b_col2:
        batch_thr = st.number_input(
            "Ngưỡng phân loại (t):",
            min_value=0.05,
            max_value=0.95,
            value=float(best_threshold if batch_model == best_model_name else 0.50),
            step=0.01,
            key="batch_thr_input",
        )
    with b_col3:
        sample_template = df_test[INPUT_COLS_RAW].head(20).to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ Tải file CSV mẫu (20 dòng chuẩn định dạng)",
            data=sample_template,
            file_name="mau_du_lieu_cam_bien.csv",
            mime="text/csv",
            width="stretch",
        )

    up_col, demo_col = st.columns([2.2, 1.0])
    with up_col:
        uploaded_file = st.file_uploader(
            "Chọn file CSV để tải lên:",
            type=["csv"],
            help="Chấp nhận tên cột gốc có [K]/[rpm]/[Nm]/[min] hoặc tên cột đã bỏ dấu ngoặc vuông.",
        )
    with demo_col:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        use_demo_50 = st.button("🧪 Dùng thử 50 mẫu trích từ tập Test", width="stretch", type="primary")

    if use_demo_50:
        # Trích 42 mẫu bình thường + 8 mẫu hỏng từ tập Test để minh họa trực quan
        neg_part = df_test[df_test["Machine failure"] == 0].sample(n=42, random_state=42)
        pos_part = df_test[df_test["Machine failure"] == 1].sample(n=min(8, int((df_test["Machine failure"] == 1).sum())), random_state=42)
        demo_df = pd.concat([pos_part, neg_part]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        st.session_state["batch_df_source"] = demo_df
        st.session_state["batch_source_label"] = "50 mẫu trích từ tập Kiểm thử (gồm 42 mẫu bình thường + 8 mẫu hỏng thực tế)"

    if uploaded_file is not None:
        try:
            st.session_state["batch_df_source"] = read_csv_smart(uploaded_file)
            st.session_state["batch_source_label"] = f"File đã tải lên: {uploaded_file.name}"
        except Exception as exc:
            st.error(f"Không thể đọc file CSV: {exc}")

    if "batch_df_source" in st.session_state:
        raw_batch_df: pd.DataFrame = st.session_state["batch_df_source"].copy()
        st.info(f"**Nguồn dữ liệu hiện tại:** {st.session_state.get('batch_source_label', '')} ({len(raw_batch_df):,} bản ghi)")

        try:
            norm_batch_df = normalize_input_columns(raw_batch_df)
            probs_b, preds_b, _, _ = predict_dataframe(norm_batch_df, batch_model, float(batch_thr))

            out_df = raw_batch_df.copy()
            out_df["Xác suất hỏng (%)"] = np.round(probs_b * 100.0, 2)
            out_df["Dự đoán (Nhãn)"] = preds_b
            out_df["Trạng thái chẩn đoán"] = np.where(
                preds_b == 1, "⚠️ Cảnh báo hỏng hóc", "✅ Bình thường"
            )

            n_total_b = len(out_df)
            n_fail_b = int((preds_b == 1).sum())
            n_ok_b = n_total_b - n_fail_b

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Tổng số mẫu dự đoán", f"{n_total_b:,}")
            k2.metric("Bình thường (Nhãn 0)", f"{n_ok_b:,}")
            k3.metric("Cảnh báo hỏng (Nhãn 1)", f"{n_fail_b:,}")
            k4.metric("Tỷ lệ cảnh báo", f"{(n_fail_b / n_total_b * 100):.2f}%")

            if "Machine failure" in out_df.columns:
                m_batch = compute_metrics(out_df["Machine failure"], probs_b, threshold=float(batch_thr))
                st.success(
                    f"**Đối chiếu với cột `Machine failure` có sẵn trong dữ liệu:** "
                    f"TP = `{m_batch['TP']}` | TN = `{m_batch['TN']}` | FP = `{m_batch['FP']}` | FN = `{m_batch['FN']}` "
                    f"| Precision = `{m_batch['Precision']:.4f}` | Recall = `{m_batch['Recall']:.4f}` | F1 = `{m_batch['F1-score']:.4f}`"
                )

            def highlight_predicted_failure(row: pd.Series) -> List[str]:
                if row["Dự đoán (Nhãn)"] == 1:
                    return ["background-color: #fee2e2; color: #7f1d1d; font-weight: 600;"] * len(row)
                return [""] * len(row)

            styled_out = out_df.style.apply(highlight_predicted_failure, axis=1)
            st.dataframe(styled_out, width="stretch", height=440)

            csv_bytes = out_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "💾 Tải toàn bộ bảng kết quả dự đoán (.CSV)",
                data=csv_bytes,
                file_name="ket_qua_du_doan_bao_tri.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as exc:
            st.error(f"Lỗi xử lý dữ liệu đầu vào: {exc}")


# ------------------------------------------------------------------------------
# TAB 3: QUY TRÌNH TIỀN XỬ LÝ & KHÁM PHÁ DỮ LIỆU (CHƯƠNG 2)
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Quy trình Kiểm tra Chất lượng, Tiền xử lý & Khám phá Dữ liệu (Chương 2)")

    q1, q2, q3, q4 = st.columns(4)
    with q1:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">Quy mô bộ dữ liệu</div>
            <div class="kpi-value">{quality_summary['n_rows']:,} × {quality_summary['n_cols']}</div>
            <div class="kpi-sub">AI4I 2020 (UCI Repository)</div></div>""",
            unsafe_allow_html=True,
        )
    with q2:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">Giá trị thiếu / Trùng lặp</div>
            <div class="kpi-value">{quality_summary['missing_total']} / {quality_summary['no_id_duplicates']}</div>
            <div class="kpi-sub">Dữ liệu sạch hoàn toàn (0 null, 0 trùng)</div></div>""",
            unsafe_allow_html=True,
        )
    with q3:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">Lớp thiểu số (Có hỏng = 1)</div>
            <div class="kpi-value">{quality_summary['n_pos']:,} mẫu</div>
            <div class="kpi-sub">Chiếm {quality_summary['n_pos']/quality_summary['n_rows']*100:.2f}% (Mất cân bằng 1 : 28.5)</div></div>""",
            unsafe_allow_html=True,
        )
    with q4:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">Đặc trưng đầu vào</div>
            <div class="kpi-value">6 gốc → 7 số</div>
            <div class="kpi-sub">Loại UID/Product ID & 5 nhãn rò rỉ</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("#### 1. Phương án lựa chọn thuộc tính & Chống rò rỉ nhãn (Bảng 2.2)")
    table_2_2 = pd.DataFrame([
        {
            "Nhóm thuộc tính": "Định danh bản ghi & sản phẩm",
            "Các cột": "UDI, Product ID",
            "Số lượng": 2,
            "Phương án xử lý": "Loại bỏ khỏi đầu vào (không phản ánh thông số vận hành)",
        },
        {
            "Nhóm thuộc tính": "Nhãn lỗi thành phần",
            "Các cột": "TWF, HDF, PWF, OSF, RNF",
            "Số lượng": 5,
            "Phương án xử lý": "Loại bỏ để chống rò rỉ nhãn (Label Leakage)",
        },
        {
            "Nhóm thuộc tính": "Đặc điểm sản phẩm",
            "Các cột": "Type (L, M, H)",
            "Số lượng": 1,
            "Phương án xử lý": "Giữ lại và mã hóa One-Hot thành Type_M, Type_H (tham chiếu L)",
        },
        {
            "Nhóm thuộc tính": "Thông số vận hành cảm biến",
            "Các cột": "Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]",
            "Số lượng": 5,
            "Phương án xử lý": "Giữ lại; chuẩn hóa StandardScaler (chỉ fit trên Train) cho Logistic Regression",
        },
        {
            "Nhóm thuộc tính": "Nhãn mục tiêu",
            "Các cột": "Machine failure (0 / 1)",
            "Số lượng": 1,
            "Phương án xử lý": "Tách riêng làm biến đầu ra phân loại nhị phân",
        },
    ])
    st.dataframe(table_2_2, width="stretch", hide_index=True)

    c_split, c_enc = st.columns([1.35, 1.0], gap="large")
    with c_split:
        st.markdown("#### 2. Phân chia dữ liệu phân tầng 70% – 15% – 15% (Bảng 2.3)")
        st.dataframe(df_split, width="stretch", hide_index=True)
    with c_enc:
        st.markdown("#### 3. Quy tắc mã hóa thuộc tính Type (Bảng 2.4)")
        table_2_4 = pd.DataFrame([
            {"Giá trị Type": "L (Low - Tham chiếu)", "Type_M": 0, "Type_H": 0},
            {"Giá trị Type": "M (Medium)", "Type_M": 1, "Type_H": 0},
            {"Giá trị Type": "H (High)", "Type_M": 0, "Type_H": 1},
        ])
        st.dataframe(table_2_4, width="stretch", hide_index=True)
        st.caption("Sau mã hóa + làm sạch tên cột (bỏ `[` và `]`), đầu vào mô hình gồm đúng 7 đặc trưng số.")

    st.markdown("#### 4. Thống kê mô tả 5 thông số vận hành (N = 10.000)")
    st.dataframe(df_desc, width="stretch", hide_index=True)

    st.markdown("#### 5. Trực quan hóa Phân bố Nhãn & Ma trận Tương quan (trên tập Train)")
    img_c1, img_c2 = st.columns(2, gap="large")
    with img_c1:
        st.image(str(CHARTS_DIR / "label_distribution.png"), caption="Hình: Phân bố nhãn hỏng hóc trong bộ dữ liệu AI4I 2020", width="stretch")
    with img_c2:
        st.image(str(CHARTS_DIR / "correlation_matrix.png"), caption="Hình: Ma trận tương quan các biến số trên tập Train (tránh rò rỉ dữ liệu)", width="stretch")


# ------------------------------------------------------------------------------
# TAB 4: KẾT QUẢ THỰC NGHIỆM TRAIN - VALIDATION - TEST (CHƯƠNG 3)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Kết quả Thực nghiệm Train – Validation – Test (Chương 3)")

    best_val = df_val.loc[df_val["Mô hình"] == best_model_name].iloc[0]
    test_default = df_test_summary.iloc[0]
    test_opt = df_test_summary.iloc[1]

    t_k1, t_k2, t_k3, t_k4, t_k5 = st.columns(5)
    t_k1.metric("Mô hình tốt nhất (Val AP)", best_model_name.split(" (")[0], best_model_name.split("(")[1].replace(")", ""))
    t_k2.metric("Validation AP", f"{best_val['AP']:.4f}", f"PR-AUC: {best_val['PR-AUC']:.4f}")
    t_k3.metric("Ngưỡng tối ưu t*", f"{best_threshold:.2f}", "Tối ưu F1 trên Val")
    t_k4.metric("Test F1-score (t*)", f"{test_opt['F1-score']:.4f}", f"{test_opt['F1-score'] - test_default['F1-score']:+.4f} vs t=0.50")
    t_k5.metric("Test AP / Recall (t*)", f"{test_opt['AP']:.4f}", f"Recall: {test_opt['Recall']:.4f}")

    st.markdown("#### 1. Bảng so sánh 8 kịch bản trên tập Validation (Mục 3.2.4)")
    def highlight_best_val_row(row: pd.Series) -> List[str]:
        if row["Mô hình"] == best_model_name:
            return ["background-color: #dbeafe; color: #1e3a8a; font-weight: 700;"] * len(row)
        return [""] * len(row)

    st.dataframe(df_val.style.apply(highlight_best_val_row, axis=1), width="stretch", hide_index=True)

    st.markdown("#### 2. Đánh giá Quá khớp: So sánh Train vs Validation (Mục 3.2.5)")
    def style_overfit_row(row: pd.Series) -> List[str]:
        note = str(row["Đánh giá quá khớp"])
        styles = [""] * len(row)
        idx = list(row.index).index("Đánh giá quá khớp")
        if "Kiểm soát tốt" in note:
            styles[idx] = "background-color: #dcfce7; color: #166534; font-weight: 700;"
        elif "Chênh lệch nhẹ" in note:
            styles[idx] = "background-color: #fef9c3; color: #854d0e; font-weight: 600;"
        elif "Chưa khớp" in note:
            styles[idx] = "background-color: #f1f5f9; color: #334155; font-weight: 600;"
        elif "quá khớp" in note:
            styles[idx] = "background-color: #fee2e2; color: #991b1b; font-weight: 700;"
        return styles

    st.dataframe(df_overfit.style.apply(style_overfit_row, axis=1), width="stretch", hide_index=True)

    st.markdown("#### 3. Đường cong Precision–Recall & Tối ưu Ngưỡng phân loại trên Validation")
    vc1, vc2 = st.columns(2, gap="large")
    with vc1:
        st.image(str(CHARTS_DIR / "pr_curve_validation.png"), caption="Đường cong Precision–Recall của 8 mô hình trên Validation", width="stretch")
    with vc2:
        st.image(str(CHARTS_DIR / "threshold_tuning.png"), caption=f"Biến thiên Precision, Recall, F1-score theo ngưỡng t (t* = {best_threshold:.2f})", width="stretch")

    st.markdown("#### 4. Kết quả Kiểm thử cuối cùng trên tập Test độc lập (1.500 mẫu - Mục 3.3.1)")
    st.dataframe(df_test_summary, width="stretch", hide_index=True)

    tc1, tc2 = st.columns(2, gap="large")
    with tc1:
        st.image(str(CHARTS_DIR / "confusion_matrix_test.png"), caption=f"Ma trận nhầm lẫn trên tập Test tại ngưỡng t* = {best_threshold:.2f}", width="stretch")
    with tc2:
        st.image(str(CHARTS_DIR / "feature_importance.png"), caption=f"Mức độ quan trọng của 7 đặc trưng - {best_model_name}", width="stretch")

    # Phân tích chi tiết TP / FP / FN trên Test set
    st.markdown(f"#### 5. Phân tích Khả năng phát hiện hỏng hóc, Cảnh báo nhầm (FP) và Bỏ sót (FN) tại t* = {best_threshold:.2f}")
    test_analysis_df = df_test.copy()
    test_analysis_df["Xác suất dự đoán"] = np.round(test_probs_best, 4)
    test_analysis_df["Dự đoán"] = (test_probs_best >= best_threshold).astype(int)

    tp_df = test_analysis_df[(test_analysis_df["Machine failure"] == 1) & (test_analysis_df["Dự đoán"] == 1)]
    fp_df = test_analysis_df[(test_analysis_df["Machine failure"] == 0) & (test_analysis_df["Dự đoán"] == 1)]
    fn_df = test_analysis_df[(test_analysis_df["Machine failure"] == 1) & (test_analysis_df["Dự đoán"] == 0)]
    tn_count = int(((test_analysis_df["Machine failure"] == 0) & (test_analysis_df["Dự đoán"] == 0)).sum())

    total_pos_test = len(tp_df) + len(fn_df)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Phát hiện đúng (TP)", f"{len(tp_df)} / {total_pos_test}", f"{len(tp_df)/total_pos_test*100:.1f}% ca hỏng")
    m2.metric("Đúng bình thường (TN)", f"{tn_count:,}", "Không cảnh báo nhầm")
    m3.metric("Cảnh báo nhầm (FP)", f"{len(fp_df)}", "Cần kiểm tra bổ sung")
    m4.metric("Bỏ sót hỏng hóc (FN)", f"{len(fn_df)}", f"{len(fn_df)/total_pos_test*100:.1f}% ca hỏng")

    err_c1, err_c2 = st.columns(2, gap="large")
    with err_c1:
        st.markdown(f"**Chi tiết các trường hợp Bỏ sót hỏng hóc - FN ({len(fn_df)} mẫu):**")
        if len(fn_df) > 0:
            sub_cols_present = [c for c in ["TWF", "HDF", "PWF", "OSF", "RNF"] if c in fn_df.columns]
            show_cols_fn = [get_id_col_name(fn_df), "Product ID", "Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Xác suất dự đoán"] + sub_cols_present
            st.dataframe(fn_df[show_cols_fn], width="stretch", hide_index=True)
            if "RNF" in fn_df.columns and int(fn_df["RNF"].sum()) > 0:
                st.caption(
                    f"💡 Lưu ý kỹ thuật: Trong {len(fn_df)} ca bỏ sót có {int(fn_df['RNF'].sum())} ca mang lỗi ngẫu nhiên (RNF) "
                    "hoặc lỗi xuất hiện ở vùng ranh giới không phụ thuộc trực tiếp vào cảm biến."
                )
        else:
            st.success("Không có trường hợp bỏ sót hỏng hóc nào trên tập Test!")

    with err_c2:
        st.markdown(f"**Chi tiết các trường hợp Cảnh báo nhầm - FP ({len(fp_df)} mẫu):**")
        if len(fp_df) > 0:
            show_cols_fp = [get_id_col_name(fp_df), "Product ID", "Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Xác suất dự đoán"]
            st.dataframe(fp_df[show_cols_fp], width="stretch", hide_index=True)
        else:
            st.success("Không có trường hợp cảnh báo nhầm nào trên tập Test!")
