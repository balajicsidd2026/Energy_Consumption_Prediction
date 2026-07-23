from typing import Container
import io
import json
import os
import sys
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from autogluon.tabular import TabularPredictor

# Disable GPU startup warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Add current folder to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Waste Generation Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define dataset metadata for choices
METADATA = {
  "Day_of_Week":        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
  "Peak_Season":        ["No", "Yes"],
  "Warehouse_Zone":     ["General Cargo Warehouse", "Cold Storage Warehouse", "Dangerous Goods Warehouse",
                          "Valuable Cargo Warehouse", "Express Cargo Warehouse", "Live Animal Warehouse"],
  "Warehouse_Occupancy_Percentage":  {"min": 21.64,  "max": 100.0,   "default": 62.52},
  "Employee_Count":                  {"min": 6,      "max": 45,      "default": 18},
  "Total_AWB":                       {"min": 10,     "max": 500,     "default": 132},
  "Total_Shipments":                 {"min": 11,     "max": 670,     "default": 158},
  "Total_Cargo_Weight_kg":           {"min": 1903.25, "max": 81412.23, "default": 18941.46},
  "Total_Cargo_Volume_cbm":          {"min": 7.97,   "max": 370.06,  "default": 87.17},
  "Wooden_Pallets_Handled":          {"min": 1,      "max": 262,     "default": 39},
  "Plastic_Wrapping_Used_kg":        {"min": 2.22,   "max": 324.59,  "default": 65.65},
  "Carton_Boxes_Handled":            {"min": 5,      "max": 524,     "default": 88},
  "Stretch_Film_Used_kg":            {"min": 1.1,    "max": 111.86,  "default": 34.26},
  "Damaged_Cargo_Count":             {"min": 0,      "max": 19,      "default": 5},
  "Recyclable_Waste_Percentage":     {"min": 35.02,  "max": 89.97,   "default": 65.84},
}

# ── Helper Functions ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_predictor(path: str):
    """Load the trained AutoGluon TabularPredictor."""
    try:
        return TabularPredictor.load(path)
    except AssertionError as e:
        import autogluon.tabular as _agt
        st.error(
            "**AutoGluon version mismatch.**\n\n"
            f"Installed AutoGluon version: `{_agt.__version__ if hasattr(_agt, '__version__') else 'unknown'}`\n\n"
            "The saved predictor was trained with a different AutoGluon version. "
            "Pin `autogluon.tabular` in `requirements.txt` to the exact version "
            "listed in `artifacts/models/<predictor_dir>/version.txt`, then redeploy.\n\n"
            f"Original error: {e}"
        )
        st.stop()

@st.cache_data
def load_sample_operations():
    """Load and cache the first 10,000 warehouse-day records from the dataset."""
    csv_path = "waste_generation_prediction_dataset.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, nrows=10000)
        df["display_label"] = df["Warehouse_Zone"].astype(str) + " — " + df["Date"].astype(str)
        return df
    return None

@st.cache_data
def load_full_dataset():
    csv_path = "waste_generation_prediction_dataset.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

df_samples = load_sample_operations()

# Initialize session state variables with standard defaults if not present
def init_session_state():
    defaults = {
        "Date": datetime.date.today(),
        "Warehouse_Zone": "General Cargo Warehouse",
        "Warehouse_Occupancy_Percentage": 62.52,
        "Employee_Count": 18,
        "Total_AWB": 132,
        "Total_Shipments": 158,
        "Total_Cargo_Weight_kg": 18941.46,
        "Total_Cargo_Volume_cbm": 87.17,
        "Wooden_Pallets_Handled": 39,
        "Plastic_Wrapping_Used_kg": 65.65,
        "Carton_Boxes_Handled": 88,
        "Stretch_Film_Used_kg": 34.26,
        "Damaged_Cargo_Count": 5,
        "Recyclable_Waste_Percentage": 65.84,
        "Day_of_Week": "Monday",
        "Peak_Season": "No",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

def on_operation_change():
    selected_label = st.session_state.selected_operation_label
    if df_samples is not None and selected_label:
        row = df_samples[df_samples["display_label"] == selected_label]
        if not row.empty:
            record = row.iloc[0]

            def _get(col, default=None, cast=None):
                """Safely extract a column value from the record."""
                if col in record.index and not pd.isna(record[col]):
                    val = record[col]
                    if cast == "int":
                        return int(val)
                    elif cast == "float":
                        return float(val)
                    elif cast == "str":
                        return str(val)
                    return val
                return default

            # Explicitly extract every field from the fetched row
            st.session_state.Warehouse_Zone                    = _get("Warehouse_Zone", "General Cargo Warehouse", "str")
            st.session_state.Warehouse_Occupancy_Percentage    = _get("Warehouse_Occupancy_Percentage", 62.52, "float")
            st.session_state.Employee_Count                    = _get("Employee_Count", 18, "int")
            st.session_state.Total_AWB                         = _get("Total_AWB", 132, "int")
            st.session_state.Total_Shipments                   = _get("Total_Shipments", 158, "int")
            st.session_state.Total_Cargo_Weight_kg              = _get("Total_Cargo_Weight_kg", 18941.46, "float")
            st.session_state.Total_Cargo_Volume_cbm             = _get("Total_Cargo_Volume_cbm", 87.17, "float")
            st.session_state.Wooden_Pallets_Handled             = _get("Wooden_Pallets_Handled", 39, "int")
            st.session_state.Plastic_Wrapping_Used_kg           = _get("Plastic_Wrapping_Used_kg", 65.65, "float")
            st.session_state.Carton_Boxes_Handled               = _get("Carton_Boxes_Handled", 88, "int")
            st.session_state.Stretch_Film_Used_kg               = _get("Stretch_Film_Used_kg", 34.26, "float")
            st.session_state.Damaged_Cargo_Count                = _get("Damaged_Cargo_Count", 5, "int")
            st.session_state.Recyclable_Waste_Percentage        = _get("Recyclable_Waste_Percentage", 65.84, "float")
            st.session_state.Day_of_Week                        = _get("Day_of_Week", "Monday", "str")
            st.session_state.Peak_Season                        = _get("Peak_Season", "No", "str")

            raw_date = _get("Date", None, "str")
            if raw_date:
                try:
                    st.session_state.Date = pd.to_datetime(raw_date).date()
                except Exception:
                    pass

            for k in ["Warehouse_Occupancy_Percentage", "Employee_Count", "Total_AWB", "Total_Shipments",
                      "Total_Cargo_Weight_kg", "Total_Cargo_Volume_cbm", "Wooden_Pallets_Handled",
                      "Plastic_Wrapping_Used_kg", "Carton_Boxes_Handled", "Stretch_Film_Used_kg",
                      "Damaged_Cargo_Count", "Recyclable_Waste_Percentage"]:
                st.session_state[f"{k}_num"] = st.session_state[k]

# ── Custom CSS for Premium UI ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif !important;
}

.info-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.015);
    text-align: left;
}
.info-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.info-value {
    font-size: 1.95rem;
    font-weight: 700;
    color: #1e293b;
}

.info-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 2rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.015);
    text-align: left;
}
.info-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #3b82f6;
    margin-bottom: 1.2rem;
}
.info-text {
    font-size: 1.05rem;
    color: #475569;
    line-height: 1.65;
}
.info-list {
    font-size: 1.05rem;
    color: #475569;
    line-height: 1.85;
    padding-left: 1.5rem;
    margin: 0;
}
.info-list li {
    margin-bottom: 0.5rem;
}

button[data-baseweb="tab"] {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #4b5563 !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s ease-in-out !important;
    padding: 10px 20px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #ff4b4b !important;
    border-bottom: 2px solid #ff4b4b !important;
}
button[data-baseweb="tab"]:hover {
    color: #ff4b4b !important;
}

.stButton > button {
    background-color : #1565ff;
    color            : white;
    border-radius    : 10px;
    height           : 52px;
    width            : 100%;
    font-size        : 17px;
    font-weight      : bold;
    border           : none;
    margin-top       : 4px;
    transition       : background 0.2s;
}
.stButton > button:hover { background-color: #0d4ed8; }
</style>
""", unsafe_allow_html=True)

# Try to load optimized model clone first, fall back to original predictor
PREDICTOR_PATH = "artifacts/models/demo_way3_clone_for_deployment"
if not os.path.exists(PREDICTOR_PATH):
    PREDICTOR_PATH = "artifacts/models/autogluon_predictor"
EVAL_REPORT_PATH = "artifacts/evaluation/evaluation_report.json"
LEADERBOARD_PATH = "artifacts/evaluation/final_leaderboard.csv"
FEAT_IMP_PATH = "artifacts/evaluation/feature_importance.csv"

# ── Page Header ───────────────────────────────────────────────────────────────
st.title("Waste Generation Prediction")

# ── Main Tabbed Interface ─────────────────────────────────────────────────────
tab_overview, tab_pred= st.tabs(
    ["Overview", "Prediction"]
)

# ── Tab: Overview ─────────────────────────────────────────────────────────────
with tab_overview:
    st.markdown("""
    <div style="display: flex; gap: 1.5rem; margin-bottom: 2rem;">
        <div class="info-card">
            <div class="info-label">Domain</div>
            <div class="info-value">Airport Cargo Warehouse Sustainability</div>
        </div>
        <div class="info-card">
            <div class="info-label">User Type</div>
            <div class="info-value">Warehouse / Sustainability Manager</div>
        </div>
        <div class="info-card">
            <div class="info-label">Target Variable</div>
            <div class="info-value">Waste Generated (kg)</div>
        </div>
    </div>
    
    <div class="info-box" style="margin-bottom: 1.5rem;">
        <div class="info-header">Use Case Overview</div>
        <div class="info-text">
            The Waste Generation Prediction platform forecasts the daily waste generated inside
            each cargo warehouse at King Abdulaziz International Airport (Jeddah). Using
            operational drivers — warehouse zone, occupancy, staffing, shipment and cargo
            volumes, packaging material usage (wooden pallets, plastic wrapping, carton boxes,
            stretch film), damaged cargo counts, recyclable-waste share, and seasonality — the
            model estimates expected warehouse waste per day. This considers warehouse
            operations only, and excludes passenger terminal, aircraft, catering, external
            transportation, and office waste. This enables warehouse and sustainability teams
            to plan waste management capacity, flag abnormal generation, and identify recycling
            opportunities across the cargo handling network.
        </div>
    </div>
    
    <div style="display: flex; gap: 1.5rem; margin-bottom: 2rem;">
        <div class="info-box" style="flex: 1;">
            <div class="info-header">Business Objectives</div>
                <ul class="info-list">
                <li>Forecast warehouse-level waste generation accurately</li>
                <li>Identify high-waste drivers (packaging material usage, cargo volume)</li>
                <li>Support waste budgeting and sustainability reporting</li>
                <li>Detect abnormal waste generation patterns early</li>
                <li>Optimize packaging material usage and recycling programs</li>
                <li>Reduce operational waste disposal costs</li>
            </ul>
        </div>
        <div class="info-box" style="flex: 1;">
            <div class="info-header">Key Benefits</div>
            <ul class="info-list">
                <li>Data-driven waste budgeting per warehouse zone</li>
                <li>Early detection of inefficient packaging practices</li>
                <li>Better alignment of recycling programs with demand</li>
                <li>Lower waste disposal costs and environmental footprint</li>
                <li>Improved sustainability reporting accuracy</li>
                <li>Actionable feature-level insights for warehouse teams</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Tab: Prediction ───────────────────────────────────────────────────────────
with tab_pred:
    if not os.path.exists(PREDICTOR_PATH):
        st.error(
            "**Predictor files not found!**\n\n"
            "Please run the training pipeline first to build the model:\n"
            "```bash\npy run_pipeline.py\n```"
        )
    else:
        with st.spinner("Initializing AutoGluon Predictor..."):
            predictor = load_predictor(PREDICTOR_PATH)

        tab1, tab2 = st.tabs(["Single Prediction", "Batch CSV Upload"])

        # ── Tab 1: Single Prediction ──────────────────────────────────────────
        with tab1:
            inf_mode = st.radio(
                "Input Method:",
                options=["Manual Entry", "Fetch by Warehouse / Date"],
                key="inference_mode",
                horizontal=True
            )
            
            if inf_mode == "Fetch by Warehouse / Date" and df_samples is not None:
                st.selectbox(
                    "Search & Select Warehouse-Day Record",
                    options=df_samples["display_label"].tolist(),
                    key="selected_operation_label",
                    on_change=on_operation_change
                )
            st.subheader("Warehouse & Operational Details")
            
            is_disabled = (inf_mode == "Fetch by Warehouse / Date")

            col1, col2 = st.columns(2)

            with col1:
                op_date = st.date_input("Operation Date", value=st.session_state.Date,
                                         disabled=is_disabled, key="Date")
                if inf_mode == "Manual Entry" and op_date is not None:
                    op_dt = pd.to_datetime(op_date)
                    st.session_state.Day_of_Week = op_dt.strftime("%A")
                    st.session_state.Peak_Season = "Yes" if op_dt.month in [5, 6, 7, 12] else "No"
                wh_zone = st.selectbox("Warehouse Zone", METADATA["Warehouse_Zone"],
                    index=METADATA["Warehouse_Zone"].index(st.session_state.Warehouse_Zone) if st.session_state.Warehouse_Zone in METADATA["Warehouse_Zone"] else 0,
                    disabled=is_disabled, key="Warehouse_Zone")

                occupancy = st.slider("Warehouse Occupancy (%)", float(METADATA["Warehouse_Occupancy_Percentage"]["min"]), float(METADATA["Warehouse_Occupancy_Percentage"]["max"]),
                    value=float(st.session_state.Warehouse_Occupancy_Percentage), disabled=is_disabled, key="Warehouse_Occupancy_Percentage_num")
                st.session_state.Warehouse_Occupancy_Percentage = occupancy

                employees = st.number_input("Employee Count",
                    min_value=int(METADATA["Employee_Count"]["min"]), max_value=int(METADATA["Employee_Count"]["max"]),
                    value=int(st.session_state.Employee_Count), step=1, disabled=is_disabled, key="Employee_Count_num")
                st.session_state.Employee_Count = employees

                total_awb = st.number_input("Total AWB",
                    min_value=int(METADATA["Total_AWB"]["min"]), max_value=int(METADATA["Total_AWB"]["max"]),
                    value=int(st.session_state.Total_AWB), step=1, disabled=is_disabled, key="Total_AWB_num")
                st.session_state.Total_AWB = total_awb

                shipments = st.number_input("Total Shipments",
                    min_value=int(METADATA["Total_Shipments"]["min"]), max_value=int(METADATA["Total_Shipments"]["max"]),
                    value=int(st.session_state.Total_Shipments), step=1, disabled=is_disabled, key="Total_Shipments_num")
                st.session_state.Total_Shipments = shipments

                cargo_wt = st.number_input("Total Cargo Weight (kg)",
                    min_value=float(METADATA["Total_Cargo_Weight_kg"]["min"]), max_value=float(METADATA["Total_Cargo_Weight_kg"]["max"]),
                    value=float(st.session_state.Total_Cargo_Weight_kg), step=100.0, disabled=is_disabled, key="Total_Cargo_Weight_kg_num")
                st.session_state.Total_Cargo_Weight_kg = cargo_wt

                cargo_vol = st.number_input("Total Cargo Volume (cbm)",
                    min_value=float(METADATA["Total_Cargo_Volume_cbm"]["min"]), max_value=float(METADATA["Total_Cargo_Volume_cbm"]["max"]),
                    value=float(st.session_state.Total_Cargo_Volume_cbm), step=1.0, disabled=is_disabled, key="Total_Cargo_Volume_cbm_num")
                st.session_state.Total_Cargo_Volume_cbm = cargo_vol

            with col2:
                wooden_pallets = st.number_input("Wooden Pallets Handled",
                    min_value=int(METADATA["Wooden_Pallets_Handled"]["min"]), max_value=int(METADATA["Wooden_Pallets_Handled"]["max"]),
                    value=int(st.session_state.Wooden_Pallets_Handled), step=1, disabled=is_disabled, key="Wooden_Pallets_Handled_num")
                st.session_state.Wooden_Pallets_Handled = wooden_pallets

                plastic_wrap = st.number_input("Plastic Wrapping Used (kg)",
                    min_value=float(METADATA["Plastic_Wrapping_Used_kg"]["min"]), max_value=float(METADATA["Plastic_Wrapping_Used_kg"]["max"]),
                    value=float(st.session_state.Plastic_Wrapping_Used_kg), step=1.0, disabled=is_disabled, key="Plastic_Wrapping_Used_kg_num")
                st.session_state.Plastic_Wrapping_Used_kg = plastic_wrap

                carton_boxes = st.number_input("Carton Boxes Handled",
                    min_value=int(METADATA["Carton_Boxes_Handled"]["min"]), max_value=int(METADATA["Carton_Boxes_Handled"]["max"]),
                    value=int(st.session_state.Carton_Boxes_Handled), step=1, disabled=is_disabled, key="Carton_Boxes_Handled_num")
                st.session_state.Carton_Boxes_Handled = carton_boxes

                stretch_film = st.number_input("Stretch Film Used (kg)",
                    min_value=float(METADATA["Stretch_Film_Used_kg"]["min"]), max_value=float(METADATA["Stretch_Film_Used_kg"]["max"]),
                    value=float(st.session_state.Stretch_Film_Used_kg), step=0.5, disabled=is_disabled, key="Stretch_Film_Used_kg_num")
                st.session_state.Stretch_Film_Used_kg = stretch_film

                damaged_cargo = st.number_input("Damaged Cargo Count",
                    min_value=int(METADATA["Damaged_Cargo_Count"]["min"]), max_value=int(METADATA["Damaged_Cargo_Count"]["max"]),
                    value=int(st.session_state.Damaged_Cargo_Count), step=1, disabled=is_disabled, key="Damaged_Cargo_Count_num")
                st.session_state.Damaged_Cargo_Count = damaged_cargo

                recyclable_pct = st.slider("Recyclable Waste (%)", float(METADATA["Recyclable_Waste_Percentage"]["min"]), float(METADATA["Recyclable_Waste_Percentage"]["max"]),
                    value=float(st.session_state.Recyclable_Waste_Percentage), disabled=is_disabled, key="Recyclable_Waste_Percentage_num")
                st.session_state.Recyclable_Waste_Percentage = recyclable_pct

                day_of_week = st.session_state.Day_of_Week
                peak_season = st.session_state.Peak_Season

            st.markdown("---")
            if st.button("Predict Waste Generated", type="primary", use_container_width=True):
                dt = pd.to_datetime(op_date)
                input_data = {
                    "Date": [dt],
                    "Day_of_Week": [day_of_week],
                    "Month": [dt.month],
                    "Year": [dt.year],
                    "Quarter": [dt.quarter],
                    "Week_Number": [int(dt.isocalendar().week)],
                    "Is_Weekend": [int(dt.dayofweek >= 5)],
                    "Peak_Season": [peak_season],
                    "Warehouse_Zone": [wh_zone],
                    "Warehouse_Occupancy_Percentage": [float(occupancy)],
                    "Employee_Count": [int(employees)],
                    "Total_AWB": [int(total_awb)],
                    "Total_Shipments": [int(shipments)],
                    "Total_Cargo_Weight_kg": [float(cargo_wt)],
                    "Total_Cargo_Volume_cbm": [float(cargo_vol)],
                    "Wooden_Pallets_Handled": [int(wooden_pallets)],
                    "Plastic_Wrapping_Used_kg": [float(plastic_wrap)],
                    "Carton_Boxes_Handled": [int(carton_boxes)],
                    "Stretch_Film_Used_kg": [float(stretch_film)],
                    "Damaged_Cargo_Count": [int(damaged_cargo)],
                    "Recyclable_Waste_Percentage": [float(recyclable_pct)],
                }
                input_df = pd.DataFrame(input_data)

                with st.spinner("Estimating waste generation..."):
                    predicted_kg = float(predictor.predict(input_df).values[0])

                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Predicted Waste Generated", f"{predicted_kg:.2f} kg")
                col_res2.metric("Warehouse Zone", f"{wh_zone}")
                col_res3.metric("Occupancy", f"{occupancy:.1f}%")


        # ── Tab 2: Batch CSV Upload ───────────────────────────────────────────
        with tab2:
            st.subheader("Upload Batch CSV for Predictions")
            
            template_path = "sample_batch_test.csv"
            if os.path.exists(template_path):
                try:
                    df_template = pd.read_csv(template_path)
                    if "Waste_Generated_kg" in df_template.columns:
                        df_template = df_template.drop(columns=["Waste_Generated_kg"])
                    template_csv = df_template.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Sample Batch Template CSV",
                        data=template_csv,
                        file_name="waste_batch_template.csv",
                        mime="text/csv",
                        help="Use this template to format your warehouse operations data correctly before uploading."
                    )
                except Exception as e:
                    pass
            
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    st.write(f"**Uploaded:** `{uploaded_file.name}` | Rows: **{len(df_upload):,}**")
                    
                    REQUIRED_COLS = [
                        "Date", "Day_of_Week", "Month", "Year", "Quarter", "Peak_Season",
                        "Warehouse_Zone", "Warehouse_Occupancy_Percentage", "Employee_Count",
                        "Total_AWB", "Total_Shipments", "Total_Cargo_Weight_kg", "Total_Cargo_Volume_cbm",
                        "Wooden_Pallets_Handled", "Plastic_Wrapping_Used_kg", "Carton_Boxes_Handled",
                        "Stretch_Film_Used_kg", "Damaged_Cargo_Count", "Recyclable_Waste_Percentage"
                    ]
                    missing_cols = [c for c in REQUIRED_COLS if c not in df_upload.columns]
                    if missing_cols:
                        st.error(f"Missing required columns: `{'`, `'.join(missing_cols)}`")
                        st.info("Please ensure your CSV contains all required model features. Use `sample_batch_test.csv` as a template.")
                    else:
                        if st.button("Run Batch Prediction", type="primary", use_container_width=True):
                            with st.spinner("Estimating waste generation for all rows..."):
                                df_pred = df_upload.copy()

                                # Derive Week_Number / Is_Weekend from Date if not already
                                # supplied — these are engineered features the model was
                                # trained on (see src/data_ingestion.py).
                                _dt = pd.to_datetime(df_pred["Date"], errors="coerce")
                                if "Week_Number" not in df_pred.columns:
                                    df_pred["Week_Number"] = _dt.dt.isocalendar().week.astype(int)
                                if "Is_Weekend" not in df_pred.columns:
                                    df_pred["Is_Weekend"] = (_dt.dt.dayofweek >= 5).astype(int)

                                num_cols = ["Employee_Count", "Total_AWB", "Total_Shipments",
                                            "Wooden_Pallets_Handled", "Carton_Boxes_Handled", "Damaged_Cargo_Count"]
                                for col in num_cols:
                                    if col in df_pred.columns:
                                        df_pred[col] = pd.to_numeric(df_pred[col], errors="coerce").fillna(0).astype(int)
                                float_cols = ["Warehouse_Occupancy_Percentage", "Total_Cargo_Weight_kg", "Total_Cargo_Volume_cbm",
                                              "Plastic_Wrapping_Used_kg", "Stretch_Film_Used_kg", "Recyclable_Waste_Percentage"]
                                for col in float_cols:
                                    if col in df_pred.columns:
                                        df_pred[col] = pd.to_numeric(df_pred[col], errors="coerce").astype(float)

                                for col in df_pred.select_dtypes(include=["object"]).columns:
                                    df_pred[col] = df_pred[col].astype(str).str.strip()

                                preds = predictor.predict(df_pred).values
                                df_pred["Predicted_Waste_Generated_kg"] = np.round(preds, 2)
                                
                                st.session_state["batch_results"] = df_pred

                    if "batch_results" in st.session_state:
                        df_res = st.session_state["batch_results"]
                        
                        total    = len(df_res)
                        avg_pred = float(df_res["Predicted_Waste_Generated_kg"].mean())
                        max_pred = float(df_res["Predicted_Waste_Generated_kg"].max())
                        min_pred = float(df_res["Predicted_Waste_Generated_kg"].min())

                        st.markdown("---")
                        st.markdown("### Prediction Summary")
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Total Records",       f"{total:,}")
                        s2.metric("Avg Predicted kg",    f"{avg_pred:.2f}")
                        s3.metric("Max Predicted kg",    f"{max_pred:.2f}")
                        s4.metric("Min Predicted kg",    f"{min_pred:.2f}")

                        st.markdown("### Filter & Search Results")
                        fc1 = st.columns(1)[0]
                        with fc1:
                            search_term = st.text_input(
                                "Search by Warehouse Zone",
                                placeholder="e.g. General Cargo Warehouse",
                                key="batch_search"
                            )
                        
                        df_view = df_res.copy()
                        if search_term.strip():
                            term = search_term.strip().lower()
                            if "Warehouse_Zone" in df_view.columns:
                                df_view = df_view[df_view["Warehouse_Zone"].astype(str).str.lower().str.contains(term, na=False)]
                        
                        df_view = df_view.sort_values(by="Predicted_Waste_Generated_kg", ascending=False)
                        
                        st.markdown(f"**Showing {len(df_view):,} of {total:,} records**")
                        
                        id_cols   = [c for c in ["Warehouse_Zone", "Date"] if c in df_view.columns]
                        pred_cols = ["Predicted_Waste_Generated_kg"]
                        other_cols = [c for c in df_view.columns if c not in pred_cols + id_cols]
                        display_cols = pred_cols + id_cols + other_cols
                        
                        st.dataframe(
                            df_view[display_cols].reset_index(drop=True),
                            use_container_width=True,
                            height=500
                        )
                        
                        st.markdown("---")
                        csv_bytes = df_res.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Full Results CSV",
                            data=csv_bytes,
                            file_name="waste_generation_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
