"""
Huấn luyện, xác thực, tối ưu ngưỡng và kiểm thử các mô hình học máy
cho bài toán: "Dự đoán hỏng hóc máy phục vụ bảo trì dự đoán trên dữ liệu cảm biến mất cân bằng"
Bộ dữ liệu: AI4I 2020 Predictive Maintenance Dataset (UCI)
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Cấu hình đường dẫn & hằng số
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "ai4i2020.csv"
RESULTS_DIR = BASE_DIR / "results"
CHARTS_DIR = BASE_DIR / "charts"
ARTIFACTS_PATH = BASE_DIR / "model_artifacts.pkl"

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
RANDOM_STATE = 42

NUMERIC_COLS_RAW: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
NUMERIC_COLS_CLEAN: List[str] = [
    col.replace("[", "").replace("]", "") for col in NUMERIC_COLS_RAW
]
INPUT_COLS_RAW: List[str] = ["Type"] + NUMERIC_COLS_RAW
FEATURE_COLS: List[str] = NUMERIC_COLS_CLEAN + ["Type_M", "Type_H"]
TARGET_COL = "Machine failure"
FAILURE_SUB_COLS: List[str] = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Cấu hình font matplotlib hỗ trợ tiếng Việt
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.unicode_minus"] = False


def read_csv_smart(source: Any) -> pd.DataFrame:
    """Đọc CSV tự động nhận diện dấu phân cách ',' hoặc ';' và loại bỏ khoảng trắng thừa ở tên cột."""
    df_loaded = pd.read_csv(source, sep=None, engine="python", encoding="utf-8-sig")
    df_loaded.columns = [str(c).strip() for c in df_loaded.columns]
    return df_loaded


def download_dataset_if_needed(data_path: Path = DATA_PATH) -> Path:
    """Kiểm tra file ai4i2020.csv và tự động tải từ UCI nếu chưa tồn tại."""
    if data_path.exists() and data_path.stat().st_size > 0:
        print(f"[INFO] Đã tìm thấy bộ dữ liệu tại: {data_path}")
        return data_path

    print(f"[INFO] Chưa có {data_path.name}. Đang tải tự động từ UCI Repository...")
    urllib.request.urlretrieve(UCI_URL, data_path)
    print(f"[INFO] Tải dữ liệu thành công: {data_path} ({data_path.stat().st_size:,} bytes)")
    return data_path


def normalize_input_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa tên cột đầu vào để chấp nhận cả tên gốc (có ngoặc vuông)
    hoặc tên đã làm sạch (không có ngoặc vuông).
    """
    df_out = df_in.copy()
    rename_map = {}
    for raw_col, clean_col in zip(NUMERIC_COLS_RAW, NUMERIC_COLS_CLEAN):
        if clean_col in df_out.columns and raw_col not in df_out.columns:
            rename_map[clean_col] = raw_col
    if rename_map:
        df_out = df_out.rename(columns=rename_map)
    return df_out


def encode_and_clean_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Mục 2.2.4:
    - Mã hóa One-Hot thuộc tính Type với L làm mức tham chiếu (Type_M, Type_H).
    - Làm sạch tên 5 cột số (bỏ '[' và ']').
    - Trả về đúng 7 đặc trưng số theo thứ tự cố định.
    """
    df_norm = normalize_input_columns(df_in)
    missing = [c for c in INPUT_COLS_RAW if c not in df_norm.columns]
    if missing:
        raise ValueError(f"Thiếu các cột đầu vào bắt buộc: {missing}")

    type_series = df_norm["Type"].astype(str).str.strip().str.upper()
    invalid_types = set(type_series.unique()) - {"L", "M", "H"}
    if invalid_types:
        raise ValueError(f"Giá trị cột Type không hợp lệ: {invalid_types}. Chỉ chấp nhận L, M, H.")

    encoded = pd.DataFrame(index=df_norm.index)
    for raw_col, clean_col in zip(NUMERIC_COLS_RAW, NUMERIC_COLS_CLEAN):
        encoded[clean_col] = pd.to_numeric(df_norm[raw_col], errors="raise").astype(float)

    encoded["Type_M"] = (type_series == "M").astype(int)
    encoded["Type_H"] = (type_series == "H").astype(int)
    return encoded[FEATURE_COLS]


def apply_scaler(
    X_encoded: pd.DataFrame, scaler: StandardScaler, fit: bool = False
) -> pd.DataFrame:
    """
    Chuẩn hóa CHỈ 5 thuộc tính vận hành bằng StandardScaler;
    giữ nguyên 2 biến nhị phân Type_M và Type_H.
    """
    X_scaled = X_encoded.copy()
    if fit:
        X_scaled[NUMERIC_COLS_CLEAN] = scaler.fit_transform(X_encoded[NUMERIC_COLS_CLEAN])
    else:
        X_scaled[NUMERIC_COLS_CLEAN] = scaler.transform(X_encoded[NUMERIC_COLS_CLEAN])
    return X_scaled


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Tính đầy đủ Precision, Recall, F1, AP, PR-AUC (hình thang) và Ma trận nhầm lẫn."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    ap = float(average_precision_score(y_true_arr, y_prob))

    prec_curve, rec_curve, _ = precision_recall_curve(y_true_arr, y_prob)
    pr_auc = float(auc(rec_curve, prec_curve))

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "AP": ap,
        "PR-AUC": pr_auc,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def evaluate_overfitting_note(train_ap: float, val_ap: float, train_f1: float, val_f1: float) -> str:
    """Chuẩn hóa đánh giá quá khớp dựa trên Train AP, Val AP và Chênh lệch AP."""
    ap_gap = train_ap - val_ap
    if train_ap < 0.55 and val_ap < 0.50:
        return "Chưa khớp (Hạn chế tuyến tính)"
    if ap_gap <= 0.10 and val_ap >= 0.65:
        return "Kiểm soát tốt (Không quá khớp)"
    if ap_gap <= 0.15:
        return "Chênh lệch nhẹ (Chấp nhận được)"
    return "Có dấu hiệu quá khớp"


def save_charts(
    df_full: pd.DataFrame,
    X_train_raw: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    val_probs_map: Dict[str, np.ndarray],
    val_df: pd.DataFrame,
    best_model_name: str,
    best_model_obj: Any,
    best_threshold: float,
    threshold_scan_df: pd.DataFrame,
    test_probs: np.ndarray,
) -> None:
    """Vẽ và lưu toàn bộ 6 biểu đồ sắc nét (300 DPI) vào thư mục charts/."""
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Phân bố nhãn hỏng hóc (toàn bộ bộ dữ liệu)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    counts = df_full[TARGET_COL].value_counts().sort_index()
    total = len(df_full)
    labels = ["Không hỏng (Nhãn 0)", "Có hỏng (Nhãn 1)"]
    colors = ["#2563eb", "#ea580c"]
    bars = ax.bar(labels, [counts[0], counts[1]], color=colors, width=0.52, edgecolor="#1e293b", linewidth=0.8)
    for bar, count in zip(bars, [counts[0], counts[1]]):
        pct = count / total * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.015,
            f"{count:,} mẫu\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
            color="#0f172a",
        )
    ax.set_ylim(0, max(counts) * 1.16)
    ax.set_ylabel("Số lượng mẫu", fontsize=11)
    ax.set_title("Phân bố nhãn hỏng hóc trong bộ dữ liệu AI4I 2020 (N = 10.000)", fontsize=12.5, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "label_distribution.png", dpi=300)
    plt.close(fig)

    # 2. Ma trận tương quan các biến số trên tập Train (tránh rò rỉ thông tin Val/Test)
    train_corr_df = X_train_raw[NUMERIC_COLS_CLEAN].copy()
    train_corr_df["Machine failure"] = y_train.values
    short_names = {
        "Air temperature K": "Air temp [K]",
        "Process temperature K": "Process temp [K]",
        "Rotational speed rpm": "Speed [rpm]",
        "Torque Nm": "Torque [Nm]",
        "Tool wear min": "Tool wear [min]",
        "Machine failure": "Machine failure",
    }
    train_corr_df = train_corr_df.rename(columns=short_names)
    corr_matrix = train_corr_df.corr()

    fig, ax = plt.subplots(figsize=(8.2, 6.5))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.6,
        cbar_kws={"shrink": 0.82},
        ax=ax,
    )
    ax.set_title("Ma trận tương quan các biến số và nhãn mục tiêu trên tập Train", fontsize=12.5, fontweight="bold", pad=12)
    plt.xticks(rotation=25, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "correlation_matrix.png", dpi=300)
    plt.close(fig)

    # 3. Đường cong Precision-Recall trên tập Validation
    fig, ax = plt.subplots(figsize=(8.8, 6.0))
    palette = sns.color_palette("tab10", n_colors=len(val_df))
    for idx, row in val_df.iterrows():
        m_name = row["Mô hình"]
        y_prob = val_probs_map[m_name]
        prec_c, rec_c, _ = precision_recall_curve(y_val, y_prob)
        lw = 2.6 if m_name == best_model_name else 1.5
        alpha = 1.0 if m_name == best_model_name else 0.78
        linestyle = "-" if "Weighted" in m_name else "--"
        ax.plot(
            rec_c,
            prec_c,
            label=f"{m_name} (AP = {row['AP']:.4f})",
            color=palette[idx % len(palette)],
            linewidth=lw,
            linestyle=linestyle,
            alpha=alpha,
        )

    pos_rate = float(np.mean(y_val))
    ax.axhline(pos_rate, color="#64748b", linestyle=":", linewidth=1.3, label=f"Đường cơ sở ngẫu nhiên ({pos_rate:.4f})")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Recall (Độ bao phủ)", fontsize=11)
    ax.set_ylabel("Precision (Độ chính xác dự đoán dương)", fontsize=11)
    ax.set_title("Đường cong Precision–Recall của 8 mô hình trên tập Validation", fontsize=12.5, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "pr_curve_validation.png", dpi=300)
    plt.close(fig)

    # 4. Biến thiên Precision, Recall, F1-score theo ngưỡng phân loại t trên Validation
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.plot(threshold_scan_df["Threshold"], threshold_scan_df["Precision"], label="Precision", color="#2563eb", linewidth=2.0)
    ax.plot(threshold_scan_df["Threshold"], threshold_scan_df["Recall"], label="Recall", color="#16a34a", linewidth=2.0)
    ax.plot(threshold_scan_df["Threshold"], threshold_scan_df["F1-score"], label="F1-score", color="#dc2626", linewidth=2.5)

    best_row = threshold_scan_df.loc[threshold_scan_df["Threshold"] == best_threshold].iloc[0]
    ax.axvline(best_threshold, color="#0f172a", linestyle="--", linewidth=1.6, label=f"Ngưỡng tối ưu t* = {best_threshold:.2f} (F1 = {best_row['F1-score']:.4f})")
    ax.scatter([best_threshold], [best_row["F1-score"]], color="#dc2626", s=65, zorder=5)

    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Ngưỡng phân loại (t)", fontsize=11)
    ax.set_ylabel("Giá trị chỉ số", fontsize=11)
    ax.set_title(f"Tối ưu ngưỡng phân loại trên Validation ({best_model_name})", fontsize=12.2, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9.5, framealpha=0.92)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "threshold_tuning.png", dpi=300)
    plt.close(fig)

    # 5. Ma trận nhầm lẫn trên tập Test tại ngưỡng tối ưu t*
    y_test_pred = (test_probs >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    annot_labels = np.array([
        [f"TN = {tn:,}\n(Đúng - Không hỏng)", f"FP = {fp:,}\n(Cảnh báo nhầm)"],
        [f"FN = {fn:,}\n(Bỏ sót hỏng hóc)", f"TP = {tp:,}\n(Phát hiện đúng)"],
    ])

    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    sns.heatmap(
        cm,
        annot=annot_labels,
        fmt="",
        cmap="Blues",
        cbar=True,
        linewidths=1.2,
        linecolor="white",
        xticklabels=["Dự đoán 0 (Không hỏng)", "Dự đoán 1 (Có hỏng)"],
        yticklabels=["Thực tế 0 (Không hỏng)", "Thực tế 1 (Có hỏng)"],
        annot_kws={"size": 10.5, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(f"Ma trận nhầm lẫn trên tập Test (N = {len(y_test):,}, t* = {best_threshold:.2f})\n{best_model_name}", fontsize=11.8, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "confusion_matrix_test.png", dpi=300)
    plt.close(fig)

    # 6. Mức độ quan trọng của 7 đặc trưng từ mô hình tốt nhất
    if hasattr(best_model_obj, "feature_importances_"):
        importances = np.asarray(best_model_obj.feature_importances_, dtype=float)
        imp_label = "Độ quan trọng đặc trưng (Feature Importance)"
    elif hasattr(best_model_obj, "coef_"):
        importances = np.abs(best_model_obj.coef_[0])
        imp_label = "Độ lớn hệ số tuyệt đối |w|"
    else:
        importances = np.zeros(len(FEATURE_COLS))
        imp_label = "Độ quan trọng"

    if importances.sum() > 0:
        importances_pct = importances / importances.sum() * 100.0
    else:
        importances_pct = importances

    imp_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": importances,
        "Pct": importances_pct,
    }).sort_values("Pct", ascending=True)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bars = ax.barh(imp_df["Feature"], imp_df["Pct"], color="#0284c7", edgecolor="#0f172a", linewidth=0.7)
    for bar, pct in zip(bars, imp_df["Pct"]):
        ax.text(
            bar.get_width() + 0.8,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
        )
    ax.set_xlim(0, max(imp_df["Pct"]) * 1.18 if len(imp_df) else 100)
    ax.set_xlabel("Tỷ trọng đóng góp (%)", fontsize=11)
    ax.set_title(f"{imp_label}\n{best_model_name}", fontsize=11.8, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "feature_importance.png", dpi=300)
    plt.close(fig)


def run_pipeline() -> Dict[str, Any]:
    """Chạy toàn bộ quy trình kiểm tra dữ liệu, tiền xử lý, huấn luyện, tối ưu ngưỡng và xuất kết quả."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Tải & đọc dữ liệu
    data_path = download_dataset_if_needed()
    df = read_csv_smart(data_path)

    print("\n" + "=" * 72)
    print("BƯỚC 1: KIỂM TRA CHẤT LƯỢNG & LỰA CHỌN THUỘC TÍNH (MỤC 2.2.2 - 2.2.3)")
    print("=" * 72)
    print(f"Kích thước dữ liệu: {df.shape[0]:,} bản ghi x {df.shape[1]} cột")
    print("Danh sách cột:", list(df.columns))

    id_cols = [c for c in ["UDI", "UID", "Product ID"] if c in df.columns]
    na_counts = df.isna().sum()
    full_dup = int(df.duplicated().sum())
    no_id_dup = int(df.drop(columns=id_cols).duplicated().sum())

    print(f"Tổng số giá trị thiếu (missing): {int(na_counts.sum())}")
    print(f"Số dòng trùng hoàn toàn: {full_dup}")
    print(f"Số dòng trùng khi bỏ cột định danh {id_cols}: {no_id_dup}")

    label_counts = df[TARGET_COL].value_counts().sort_index()
    print(
        f"Phân bố nhãn {TARGET_COL}: "
        f"0 (Không hỏng) = {label_counts[0]:,} ({label_counts[0]/len(df)*100:.2f}%), "
        f"1 (Có hỏng) = {label_counts[1]:,} ({label_counts[1]/len(df)*100:.2f}%)"
    )
    print(f"Loại bỏ cột định danh {id_cols} và nhãn lỗi thành phần {FAILURE_SUB_COLS} để chống rò rỉ nhãn.")

    # Lưu bảng 1: Thống kê mô tả 5 biến số
    desc_df = df[NUMERIC_COLS_RAW].describe().T.reset_index()
    desc_df = desc_df.rename(columns={
        "index": "Thuộc tính",
        "count": "Số mẫu",
        "mean": "Trung bình",
        "std": "Độ lệch chuẩn",
        "min": "Nhỏ nhất",
        "25%": "Phân vị 25%",
        "50%": "Trung vị (50%)",
        "75%": "Phân vị 75%",
        "max": "Lớn nhất",
    })
    desc_df.to_csv(RESULTS_DIR / "1_thong_ke_mo_ta.csv", index=False, encoding="utf-8-sig")

    # 2. Phân chia dữ liệu trước khi biến đổi (70% - 15% - 15%, stratify, random_state=42)
    print("\n" + "=" * 72)
    print("BƯỚC 2: PHÂN CHIA DỮ LIỆU TRAIN / VALIDATION / TEST (70% - 15% - 15%)")
    print("=" * 72)
    y = df[TARGET_COL].astype(int)
    train_idx, temp_idx = train_test_split(
        df.index, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=y.loc[temp_idx], random_state=RANDOM_STATE
    )

    df_train = df.loc[train_idx].copy()
    df_val = df.loc[val_idx].copy()
    df_test = df.loc[test_idx].copy().reset_index(drop=True)

    y_train = df_train[TARGET_COL].astype(int)
    y_val = df_val[TARGET_COL].astype(int)
    y_test = df_test[TARGET_COL].astype(int)

    split_rows = []
    for split_name, split_y in [
        ("Huấn luyện (Train - 70%)", y_train),
        ("Xác thực (Validation - 15%)", y_val),
        ("Kiểm thử (Test - 15%)", y_test),
        ("Toàn bộ (100%)", y),
    ]:
        n_total = len(split_y)
        n_0 = int((split_y == 0).sum())
        n_1 = int((split_y == 1).sum())
        split_rows.append({
            "Tập dữ liệu": split_name,
            "Tổng số mẫu": n_total,
            "Tỷ lệ (%)": round(n_total / len(y) * 100, 2),
            "Không hỏng (Nhãn 0)": n_0,
            "Có hỏng (Nhãn 1)": n_1,
            "Tỷ lệ hỏng (%)": round(n_1 / n_total * 100, 2),
        })
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(RESULTS_DIR / "2_phan_chia_du_lieu.csv", index=False, encoding="utf-8-sig")
    print(split_df.to_string(index=False))

    # 3 & 4. Mã hóa Type & Chuẩn hóa StandardScaler (fit CHỈ trên Train)
    print("\n" + "=" * 72)
    print("BƯỚC 3 & 4: MÃ HÓA ONE-HOT (TYPE) VÀ CHUẨN HÓA STANDARDSCALER")
    print("=" * 72)
    X_train_raw = encode_and_clean_features(df_train[INPUT_COLS_RAW])
    X_val_raw = encode_and_clean_features(df_val[INPUT_COLS_RAW])
    X_test_raw = encode_and_clean_features(df_test[INPUT_COLS_RAW])

    scaler = StandardScaler()
    X_train_scaled = apply_scaler(X_train_raw, scaler, fit=True)
    X_val_scaled = apply_scaler(X_val_raw, scaler, fit=False)
    X_test_scaled = apply_scaler(X_test_raw, scaler, fit=False)

    print(f"7 đặc trưng sau mã hóa: {FEATURE_COLS}")
    print("StandardScaler chỉ fit trên Train (5 cột số), giữ nguyên Type_M và Type_H.")

    # 5. Huấn luyện 4 thuật toán x 2 kịch bản = 8 mô hình
    print("\n" + "=" * 72)
    print("BƯỚC 5: HUẤN LUYỆN 8 MÔ HÌNH (4 THUẬT TOÁN x 2 CẤU HÌNH)")
    print("=" * 72)
    n_neg_train = int((y_train == 0).sum())
    n_pos_train = int((y_train == 1).sum())
    scale_pos_weight = float(n_neg_train / n_pos_train)
    print(f"Tỷ lệ mất cân bằng trên Train: {n_neg_train} / {n_pos_train} => scale_pos_weight = {scale_pos_weight:.4f}")

    model_specs: List[Dict[str, Any]] = [
        {
            "name": "Logistic Regression (Baseline)",
            "algorithm": "Logistic Regression",
            "config": "Baseline",
            "uses_scaled": True,
            "estimator": LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE),
        },
        {
            "name": "Logistic Regression (Weighted)",
            "algorithm": "Logistic Regression",
            "config": "Weighted (balanced)",
            "uses_scaled": True,
            "estimator": LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        },
        {
            "name": "Random Forest (Baseline)",
            "algorithm": "Random Forest",
            "config": "Baseline",
            "uses_scaled": False,
            "estimator": RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                min_samples_split=25,
                min_samples_leaf=12,
                max_features="sqrt",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "name": "Random Forest (Weighted)",
            "algorithm": "Random Forest",
            "config": "Weighted (balanced_subsample)",
            "uses_scaled": False,
            "estimator": RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                min_samples_split=25,
                min_samples_leaf=12,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        },
        {
            "name": "XGBoost (Baseline)",
            "algorithm": "XGBoost",
            "config": "Baseline",
            "uses_scaled": False,
            "estimator": XGBClassifier(
                n_estimators=200,
                learning_rate=0.04,
                max_depth=3,
                min_child_weight=3,
                subsample=0.75,
                colsample_bytree=0.8,
                reg_alpha=1.0,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        },
        {
            "name": "XGBoost (Weighted)",
            "algorithm": "XGBoost",
            "config": "Weighted (spw=5.0)",
            "uses_scaled": False,
            "estimator": XGBClassifier(
                n_estimators=100,
                learning_rate=0.04,
                max_depth=3,
                min_child_weight=10,
                subsample=0.75,
                colsample_bytree=0.75,
                reg_alpha=1.5,
                reg_lambda=6.0,
                scale_pos_weight=5.0,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
        },
        {
            "name": "LightGBM (Baseline)",
            "algorithm": "LightGBM",
            "config": "Baseline",
            "uses_scaled": False,
            "estimator": LGBMClassifier(
                n_estimators=140,
                learning_rate=0.04,
                max_depth=3,
                num_leaves=7,
                min_child_samples=25,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=1.0,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbose=-1,
            ),
        },
        {
            "name": "LightGBM (Weighted)",
            "algorithm": "LightGBM",
            "config": "Weighted (weight=5.0)",
            "uses_scaled": False,
            "estimator": LGBMClassifier(
                n_estimators=100,
                learning_rate=0.04,
                max_depth=3,
                num_leaves=7,
                min_child_samples=35,
                subsample=0.75,
                subsample_freq=1,
                colsample_bytree=0.75,
                reg_alpha=1.5,
                reg_lambda=6.0,
                class_weight={0: 1.0, 1: 5.0},
                random_state=RANDOM_STATE,
                n_jobs=1,
                verbose=-1,
            ),
        },
    ]

    trained_models: Dict[str, Dict[str, Any]] = {}
    val_rows: List[Dict[str, Any]] = []
    overfit_rows: List[Dict[str, Any]] = []
    val_probs_map: Dict[str, np.ndarray] = {}

    for spec in model_specs:
        name = spec["name"]
        clf = spec["estimator"]
        uses_scaled = spec["uses_scaled"]

        X_tr = X_train_scaled if uses_scaled else X_train_raw
        X_va = X_val_scaled if uses_scaled else X_val_raw

        clf.fit(X_tr, y_train)
        train_prob = clf.predict_proba(X_tr)[:, 1]
        val_prob = clf.predict_proba(X_va)[:, 1]

        train_m = compute_metrics(y_train, train_prob, threshold=0.50)
        val_m = compute_metrics(y_val, val_prob, threshold=0.50)

        val_probs_map[name] = val_prob
        trained_models[name] = {
            "model": clf,
            "uses_scaled": uses_scaled,
            "algorithm": spec["algorithm"],
            "config": spec["config"],
            "val_ap": val_m["AP"],
        }

        val_rows.append({
            "Mô hình": name,
            "Thuật toán": spec["algorithm"],
            "Cấu hình": spec["config"],
            "Precision": round(val_m["Precision"], 4),
            "Recall": round(val_m["Recall"], 4),
            "F1-score": round(val_m["F1-score"], 4),
            "AP": round(val_m["AP"], 4),
            "PR-AUC": round(val_m["PR-AUC"], 4),
            "TN": val_m["TN"],
            "FP": val_m["FP"],
            "FN": val_m["FN"],
            "TP": val_m["TP"],
        })

        overfit_rows.append({
            "Mô hình": name,
            "Train AP": round(train_m["AP"], 4),
            "Val AP": round(val_m["AP"], 4),
            "Chênh lệch AP": round(train_m["AP"] - val_m["AP"], 4),
            "Train F1": round(train_m["F1-score"], 4),
            "Val F1": round(val_m["F1-score"], 4),
            "Chênh lệch F1": round(train_m["F1-score"] - val_m["F1-score"], 4),
            "Train Recall": round(train_m["Recall"], 4),
            "Val Recall": round(val_m["Recall"], 4),
            "Đánh giá quá khớp": evaluate_overfitting_note(
                train_m["AP"], val_m["AP"], train_m["F1-score"], val_m["F1-score"]
            ),
        })

    val_results_df = pd.DataFrame(val_rows).sort_values("AP", ascending=False).reset_index(drop=True)
    overfit_df = pd.DataFrame(overfit_rows).sort_values("Val AP", ascending=False).reset_index(drop=True)

    val_results_df.to_csv(RESULTS_DIR / "3_ket_qua_validation.csv", index=False, encoding="utf-8-sig")
    overfit_df.to_csv(RESULTS_DIR / "4_danh_gia_qua_khop.csv", index=False, encoding="utf-8-sig")

    print("\nBảng so sánh 8 mô hình trên Validation (sắp xếp theo AP giảm dần):")
    print(val_results_df[["Mô hình", "Precision", "Recall", "F1-score", "AP", "PR-AUC", "FP", "FN", "TP"]].to_string(index=False))

    # 6. Lựa chọn mô hình tốt nhất theo AP trên Validation & tối ưu ngưỡng t* theo F1-score
    print("\n" + "=" * 72)
    print("BƯỚC 6: LỰA CHỌN MÔ HÌNH TỐT NHẤT & TỐI ƯU NGƯỠNG PHÂN LOẠI t*")
    print("=" * 72)
    best_model_name = str(val_results_df.loc[0, "Mô hình"])
    best_info = trained_models[best_model_name]
    best_clf = best_info["model"]
    best_val_prob = val_probs_map[best_model_name]

    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scan_rows = []
    for t in thresholds:
        m = compute_metrics(y_val, best_val_prob, threshold=float(t))
        scan_rows.append({
            "Threshold": float(t),
            "Precision": m["Precision"],
            "Recall": m["Recall"],
            "F1-score": m["F1-score"],
            "TP": m["TP"],
            "FP": m["FP"],
            "FN": m["FN"],
        })
    threshold_scan_df = pd.DataFrame(scan_rows)

    max_f1 = threshold_scan_df["F1-score"].max()
    tied_best = threshold_scan_df[np.isclose(threshold_scan_df["F1-score"], max_f1)]
    # Nếu có nhiều ngưỡng đạt cùng F1 cao nhất, chọn ngưỡng gần giữa khoảng tối ưu nhất
    mid_idx = tied_best.index[len(tied_best) // 2]
    best_threshold = float(threshold_scan_df.loc[mid_idx, "Threshold"])
    best_val_f1_opt = float(threshold_scan_df.loc[mid_idx, "F1-score"])

    print(f"Mô hình tốt nhất theo AP trên Validation: {best_model_name} (AP = {val_results_df.loc[0, 'AP']:.4f})")
    print(f"Ngưỡng phân loại tối ưu trên Validation: t* = {best_threshold:.2f} (F1-score = {best_val_f1_opt:.4f})")

    # Đánh giá cuối cùng trên tập Test độc lập (1.500 mẫu)
    X_te = X_test_scaled if best_info["uses_scaled"] else X_test_raw
    test_probs = best_clf.predict_proba(X_te)[:, 1]

    test_default_m = compute_metrics(y_test, test_probs, threshold=0.50)
    test_opt_m = compute_metrics(y_test, test_probs, threshold=best_threshold)

    test_rows = [
        {
            "Mô hình": best_model_name,
            "Kịch bản ngưỡng": "Ngưỡng mặc định (t = 0.50)",
            "Ngưỡng": 0.50,
            "Precision": round(test_default_m["Precision"], 4),
            "Recall": round(test_default_m["Recall"], 4),
            "F1-score": round(test_default_m["F1-score"], 4),
            "AP": round(test_default_m["AP"], 4),
            "PR-AUC": round(test_default_m["PR-AUC"], 4),
            "TN": test_default_m["TN"],
            "FP": test_default_m["FP"],
            "FN": test_default_m["FN"],
            "TP": test_default_m["TP"],
        },
        {
            "Mô hình": best_model_name,
            "Kịch bản ngưỡng": f"Ngưỡng tối ưu Validation (t* = {best_threshold:.2f})",
            "Ngưỡng": best_threshold,
            "Precision": round(test_opt_m["Precision"], 4),
            "Recall": round(test_opt_m["Recall"], 4),
            "F1-score": round(test_opt_m["F1-score"], 4),
            "AP": round(test_opt_m["AP"], 4),
            "PR-AUC": round(test_opt_m["PR-AUC"], 4),
            "TN": test_opt_m["TN"],
            "FP": test_opt_m["FP"],
            "FN": test_opt_m["FN"],
            "TP": test_opt_m["TP"],
        },
    ]
    test_results_df = pd.DataFrame(test_rows)
    test_results_df.to_csv(RESULTS_DIR / "5_ket_qua_test.csv", index=False, encoding="utf-8-sig")

    print("\nKết quả kiểm thử cuối cùng trên tập Test (1.500 mẫu):")
    print(test_results_df[["Kịch bản ngưỡng", "Precision", "Recall", "F1-score", "AP", "PR-AUC", "TN", "FP", "FN", "TP"]].to_string(index=False))

    # 7. Vẽ & lưu 6 biểu đồ (300 DPI) và gói artifacts
    save_charts(
        df_full=df,
        X_train_raw=X_train_raw,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        val_probs_map=val_probs_map,
        val_df=val_results_df,
        best_model_name=best_model_name,
        best_model_obj=best_clf,
        best_threshold=best_threshold,
        threshold_scan_df=threshold_scan_df,
        test_probs=test_probs,
    )

    train_stats = {
        col: {
            "mean": float(X_train_raw[col].mean()),
            "std": float(X_train_raw[col].std()),
            "min": float(df_train[raw_col].min()),
            "max": float(df_train[raw_col].max()),
            "p95": float(df_train[raw_col].quantile(0.95)),
        }
        for raw_col, col in zip(NUMERIC_COLS_RAW, NUMERIC_COLS_CLEAN)
    }

    artifacts = {
        "models": trained_models,
        "best_model_name": best_model_name,
        "best_threshold": best_threshold,
        "scaler": scaler,
        "numeric_cols_raw": NUMERIC_COLS_RAW,
        "numeric_cols_clean": NUMERIC_COLS_CLEAN,
        "input_cols_raw": INPUT_COLS_RAW,
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "failure_sub_cols": [c for c in FAILURE_SUB_COLS if c in df.columns],
        "id_cols": id_cols,
        "df_test": df_test,
        "test_probs": test_probs,
        "threshold_scan": threshold_scan_df,
        "train_stats": train_stats,
        "quality_summary": {
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "missing_total": int(na_counts.sum()),
            "full_duplicates": full_dup,
            "no_id_duplicates": no_id_dup,
            "n_neg": int(label_counts[0]),
            "n_pos": int(label_counts[1]),
            "scale_pos_weight": scale_pos_weight,
        },
    }
    joblib.dump(artifacts, ARTIFACTS_PATH)

    print("\n[SUCCESS] Đã xuất toàn bộ bảng CSV vào results/, biểu đồ vào charts/ và lưu model_artifacts.pkl!")
    return artifacts


if __name__ == "__main__":
    run_pipeline()
