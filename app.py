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
    page_title="Energy Consumption Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define dataset metadata for choices
METADATA = {
  "Warehouse_ID":       ["CW001", "CW002", "DG001", "EX001", "LA001", "VW001",
                          "WH001", "WH002", "WH003", "WH004", "WH005", "WH006"],
  "Warehouse_Type":     ["Cold", "Dangerous Goods", "Express", "General", "Live Animal", "Valuable"],
  "Weather_Condition":  ["Cloudy", "Dust Storm", "Fog", "Hot", "Rain", "Sunny"],
  "Day_of_Week":        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
  "Weekend":            ["No", "Yes"],
  "Peak_Season":        ["No", "Yes"],
  "Warehouse_Area_sq_m":               {"min": 4419,  "max": 24074, "default": 16336},
  "Temperature_C":                     {"min": 22.5,  "max": 48.2,  "default": 35.2},
  "Humidity_Percentage":               {"min": 19,    "max": 86,    "default": 56},
  "Warehouse_Occupancy_Percentage":    {"min": 20,    "max": 99,    "default": 74},
  "Total_Shipments":                   {"min": 4,     "max": 473,   "default": 213},
  "Total_Cargo_Weight_kg":             {"min": 25.98, "max": 57265.17, "default": 19734.11},
  "Total_Cargo_Volume_m3":             {"min": 0.14,  "max": 363.22,   "default": 89.36},
  "Number_of_Flights":                 {"min": 1,     "max": 30,    "default": 10},
  "Forklift_Operating_Hours":          {"min": 0.0,   "max": 12.0,  "default": 6.6},
  "Conveyor_Operating_Hours":          {"min": 0.0,   "max": 14.0,  "default": 3.6},
  "HVAC_Operating_Hours":              {"min": 0.0,   "max": 14.4,  "default": 6.4},
  "Lighting_Operating_Hours":          {"min": 8.0,   "max": 12.5,  "default": 11.0},
  "Equipment_Utilization_Percentage":  {"min": 10.0,  "max": 95.0,  "default": 60.5},
  "Staff_Count":                       {"min": 8,     "max": 85,    "default": 51},
  "Staff_Availability_Percentage":     {"min": 85.0,  "max": 99.0,  "default": 92.0},
}

WAREHOUSE_MAPPING = {
    "CW001": ("Cold", 13828),
    "CW002": ("Cold", 13143),
    "DG001": ("Dangerous Goods", 11016),
    "EX001": ("Express", 14467),
    "LA001": ("Live Animal", 11771),
    "VW001": ("Valuable", 4419),
    "WH001": ("General", 23238),
    "WH002": ("General", 18912),
    "WH003": ("General", 18204),
    "WH004": ("General", 24074),
    "WH005": ("General", 20253),
    "WH006": ("General", 20006),
}

# ── Helper Functions ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_predictor(path: str):
    """Load the trained AutoGluon TabularPredictor."""
    return TabularPredictor.load(path)

@st.cache_data
def load_sample_operations():
    """Load and cache the first 10,000 warehouse-day records from the dataset."""
    csv_path = "energy_consumption_prediction_dataset.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, nrows=10000)
        df["display_label"] = df["Warehouse_ID"].astype(str) + " — " + df["Operation_Date"].astype(str)
        return df
    return None

@st.cache_data
def load_full_dataset():
    csv_path = "energy_consumption_prediction_dataset.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return None

df_samples = load_sample_operations()

# Initialize session state variables with standard defaults if not present
def init_session_state():
    defaults = {
        "Operation_Date": datetime.date.today(),
        "Warehouse_ID": "WH001",
        "Warehouse_Type": "General",
        "Warehouse_Area_sq_m": 16336,
        "Temperature_C": 35.2,
        "Humidity_Percentage": 56,
        "Weather_Condition": "Sunny",
        "Warehouse_Occupancy_Percentage": 74,
        "Total_Shipments": 213,
        "Total_Cargo_Weight_kg": 19734.11,
        "Total_Cargo_Volume_m3": 89.36,
        "Number_of_Flights": 10,
        "Forklift_Operating_Hours": 6.6,
        "Conveyor_Operating_Hours": 3.6,
        "HVAC_Operating_Hours": 6.4,
        "Lighting_Operating_Hours": 24.0,
        "Equipment_Utilization_Percentage": 60.5,
        "Staff_Count": 51,
        "Staff_Availability_Percentage": 92.0,
        "Day_of_Week": "Monday",
        "Weekend": "No",
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
            st.session_state.Warehouse_ID                      = _get("Warehouse_ID", "WH001", "str")
            st.session_state.Warehouse_Type                    = _get("Warehouse_Type", "General", "str")
            st.session_state.Warehouse_Area_sq_m               = _get("Warehouse_Area_sq_m", 16336, "int")
            st.session_state.Temperature_C                     = _get("Temperature_C", 35.2, "float")
            st.session_state.Humidity_Percentage               = _get("Humidity_Percentage", 56, "int")
            st.session_state.Weather_Condition                 = _get("Weather_Condition", "Sunny", "str")
            st.session_state.Warehouse_Occupancy_Percentage    = _get("Warehouse_Occupancy_Percentage", 74, "int")
            st.session_state.Total_Shipments                   = _get("Total_Shipments", 213, "int")
            st.session_state.Total_Cargo_Weight_kg             = _get("Total_Cargo_Weight_kg", 19734.11, "float")
            st.session_state.Total_Cargo_Volume_m3             = _get("Total_Cargo_Volume_m3", 89.36, "float")
            st.session_state.Number_of_Flights                 = _get("Number_of_Flights", 10, "int")
            st.session_state.Forklift_Operating_Hours          = _get("Forklift_Operating_Hours", 6.6, "float")
            st.session_state.Conveyor_Operating_Hours          = _get("Conveyor_Operating_Hours", 3.6, "float")
            st.session_state.HVAC_Operating_Hours              = _get("HVAC_Operating_Hours", 6.4, "float")
            st.session_state.Lighting_Operating_Hours          = _get("Lighting_Operating_Hours", 11.0, "float")
            st.session_state.Equipment_Utilization_Percentage  = _get("Equipment_Utilization_Percentage", 60.5, "float")
            st.session_state.Staff_Count                       = _get("Staff_Count", 51, "int")
            st.session_state.Staff_Availability_Percentage     = _get("Staff_Availability_Percentage", 92.0, "float")
            st.session_state.Day_of_Week                       = _get("Day_of_Week", "Monday", "str")
            st.session_state.Weekend                           = _get("Weekend", "No", "str")
            st.session_state.Peak_Season                       = _get("Peak_Season", "No", "str")

            raw_date = _get("Operation_Date", None, "str")
            if raw_date:
                try:
                    st.session_state.Operation_Date = pd.to_datetime(raw_date).date()
                except Exception:
                    pass

            for k in ["Warehouse_Area_sq_m", "Temperature_C", "Humidity_Percentage",
                      "Warehouse_Occupancy_Percentage", "Total_Shipments", "Total_Cargo_Weight_kg",
                      "Total_Cargo_Volume_m3", "Number_of_Flights", "Forklift_Operating_Hours",
                      "Conveyor_Operating_Hours", "HVAC_Operating_Hours", "Lighting_Operating_Hours",
                      "Equipment_Utilization_Percentage", "Staff_Count", "Staff_Availability_Percentage"]:
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
st.title("Energy Consumption Prediction")

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
            <div class="info-value">Airport Cargo Operations</div>
        </div>
        <div class="info-card">
            <div class="info-label">User Type</div>
            <div class="info-value">Facilities / Energy Manager</div>
        </div>
        <div class="info-card">
            <div class="info-label">Target Variable</div>
            <div class="info-value">Energy Consumption (kWh)</div>
        </div>
    </div>
    
    <div class="info-box" style="margin-bottom: 1.5rem;">
        <div class="info-header">Use Case Overview</div>
        <div class="info-text">
            The Energy Consumption Prediction platform forecasts the daily electricity usage of
            each cargo warehouse at King Abdulaziz International Airport (Jeddah), operated by
            SAL Saudi Logistics Services. Using operational drivers — warehouse type and area,
            ambient temperature and weather, occupancy, shipment and flight volumes, equipment
            operating hours (forklifts, conveyors, HVAC, lighting), staffing levels, and
            seasonality — the model estimates expected energy draw per warehouse per day. This
            enables facilities teams to plan capacity, flag anomalous consumption, and identify
            efficiency opportunities across the cargo handling network.
        </div>
    </div>
    
    <div style="display: flex; gap: 1.5rem; margin-bottom: 2rem;">
        <div class="info-box" style="flex: 1;">
            <div class="info-header">Business Objectives</div>
                <ul class="info-list">
                <li>Forecast warehouse-level energy consumption accurately</li>
                <li>Identify high-consumption drivers (HVAC, occupancy, weather)</li>
                <li>Support energy budgeting and sustainability reporting</li>
                <li>Detect abnormal consumption patterns early</li>
                <li>Optimize equipment scheduling and staffing</li>
                <li>Reduce operational energy costs</li>
            </ul>
        </div>
        <div class="info-box" style="flex: 1;">
            <div class="info-header">Key Benefits</div>
            <ul class="info-list">
                <li>Data-driven energy budgeting per warehouse</li>
                <li>Early detection of inefficient operations</li>
                <li>Better alignment of staffing/equipment with demand</li>
                <li>Lower energy costs and carbon footprint</li>
                <li>Improved sustainability reporting accuracy</li>
                <li>Actionable feature-level insights for facilities teams</li>
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
                op_date = st.date_input("Operation Date", value=st.session_state.Operation_Date,
                                         disabled=is_disabled, key="Operation_Date")
                if inf_mode == "Manual Entry" and op_date is not None:
                    op_dt = pd.to_datetime(op_date)
                    st.session_state.Day_of_Week = op_dt.strftime("%A")
                    st.session_state.Weekend = "Yes" if st.session_state.Day_of_Week in ["Friday", "Saturday"] else "No"
                    st.session_state.Peak_Season = "Yes" if op_dt.month in [5, 6, 7, 12] else "No"
                wh_id = st.selectbox("Warehouse ID", METADATA["Warehouse_ID"],
                    index=METADATA["Warehouse_ID"].index(st.session_state.Warehouse_ID) if st.session_state.Warehouse_ID in METADATA["Warehouse_ID"] else 0,
                    disabled=is_disabled, key="Warehouse_ID")
                if inf_mode == "Manual Entry" and wh_id in WAREHOUSE_MAPPING:
                    derived_type, derived_area = WAREHOUSE_MAPPING[wh_id]
                    st.session_state.Warehouse_Type = derived_type
                    st.session_state.Warehouse_Area_sq_m = derived_area
                
                wh_type = st.selectbox("Warehouse Type", METADATA["Warehouse_Type"],
                    index=METADATA["Warehouse_Type"].index(st.session_state.Warehouse_Type) if st.session_state.Warehouse_Type in METADATA["Warehouse_Type"] else 0,
                    disabled=True)
                wh_area = st.number_input("Warehouse Area (sq m)",
                    min_value=int(METADATA["Warehouse_Area_sq_m"]["min"]), max_value=int(METADATA["Warehouse_Area_sq_m"]["max"]),
                    value=int(st.session_state.Warehouse_Area_sq_m), step=100, disabled=True)
                st.session_state.Warehouse_Area_sq_m = wh_area
                temp = st.number_input("Temperature (°C)",
                    min_value=float(METADATA["Temperature_C"]["min"]), max_value=float(METADATA["Temperature_C"]["max"]),
                    value=float(st.session_state.Temperature_C), step=0.5, disabled=is_disabled, key="Temperature_C_num")
                st.session_state.Temperature_C = temp
                
                weather = st.selectbox("Weather Condition", METADATA["Weather_Condition"],
                    index=METADATA["Weather_Condition"].index(st.session_state.Weather_Condition) if st.session_state.Weather_Condition in METADATA["Weather_Condition"] else 0,
                    disabled=is_disabled, key="Weather_Condition")

                humidity = st.slider("Humidity (%)", int(METADATA["Humidity_Percentage"]["min"]), int(METADATA["Humidity_Percentage"]["max"]),
                    value=int(st.session_state.Humidity_Percentage), disabled=is_disabled, key="Humidity_Percentage_num")
                st.session_state.Humidity_Percentage = humidity

                occupancy = st.slider("Warehouse Occupancy (%)", int(METADATA["Warehouse_Occupancy_Percentage"]["min"]), int(METADATA["Warehouse_Occupancy_Percentage"]["max"]),
                    value=int(st.session_state.Warehouse_Occupancy_Percentage), disabled=is_disabled, key="Warehouse_Occupancy_Percentage_num")
                st.session_state.Warehouse_Occupancy_Percentage = occupancy


            with col2:
                shipments = st.number_input("Total Shipments",
                    min_value=int(METADATA["Total_Shipments"]["min"]), max_value=int(METADATA["Total_Shipments"]["max"]),
                    value=int(st.session_state.Total_Shipments), step=1, disabled=is_disabled, key="Total_Shipments_num")
                st.session_state.Total_Shipments = shipments
                if inf_mode == "Manual Entry":
                    st.session_state.Number_of_Flights = max(1, int(shipments // 20))
                cargo_wt = st.number_input("Total Cargo Weight (kg)",
                    min_value=float(METADATA["Total_Cargo_Weight_kg"]["min"]), max_value=float(METADATA["Total_Cargo_Weight_kg"]["max"]),
                    value=float(st.session_state.Total_Cargo_Weight_kg), step=100.0, disabled=is_disabled, key="Total_Cargo_Weight_kg_num")
                st.session_state.Total_Cargo_Weight_kg = cargo_wt
                cargo_vol = st.number_input("Total Cargo Volume (m³)",
                    min_value=float(METADATA["Total_Cargo_Volume_m3"]["min"]), max_value=float(METADATA["Total_Cargo_Volume_m3"]["max"]),
                    value=float(st.session_state.Total_Cargo_Volume_m3), step=1.0, disabled=is_disabled, key="Total_Cargo_Volume_m3_num")
                st.session_state.Total_Cargo_Volume_m3 = cargo_vol
                n_flights = st.session_state.Number_of_Flights
                forklift_hrs = st.number_input("Forklift Operating Hours",
                    min_value=float(METADATA["Forklift_Operating_Hours"]["min"]), max_value=float(METADATA["Forklift_Operating_Hours"]["max"]),
                    value=float(st.session_state.Forklift_Operating_Hours), step=0.1, disabled=is_disabled, key="Forklift_Operating_Hours_num")
                st.session_state.Forklift_Operating_Hours = forklift_hrs
                conveyor_hrs = st.number_input("Conveyor Operating Hours",
                    min_value=float(METADATA["Conveyor_Operating_Hours"]["min"]), max_value=float(METADATA["Conveyor_Operating_Hours"]["max"]),
                    value=float(st.session_state.Conveyor_Operating_Hours), step=0.1, disabled=is_disabled, key="Conveyor_Operating_Hours_num")
                st.session_state.Conveyor_Operating_Hours = conveyor_hrs
                hvac_hrs = st.number_input("HVAC Operating Hours",
                    min_value=float(METADATA["HVAC_Operating_Hours"]["min"]), max_value=float(METADATA["HVAC_Operating_Hours"]["max"]),
                    value=float(st.session_state.HVAC_Operating_Hours), step=0.1, disabled=is_disabled, key="HVAC_Operating_Hours_num")
                st.session_state.HVAC_Operating_Hours = hvac_hrs
                if inf_mode == "Manual Entry":
                    st.session_state.Lighting_Operating_Hours = 24.0
                lighting_hrs = st.session_state.Lighting_Operating_Hours
                equip_util = st.slider("Equipment Utilization (%)", float(METADATA["Equipment_Utilization_Percentage"]["min"]), float(METADATA["Equipment_Utilization_Percentage"]["max"]),
                    value=float(st.session_state.Equipment_Utilization_Percentage), disabled=is_disabled, key="Equipment_Utilization_Percentage_num")
                st.session_state.Equipment_Utilization_Percentage = equip_util
                staff_count = st.number_input("Staff Count",
                    min_value=int(METADATA["Staff_Count"]["min"]), max_value=int(METADATA["Staff_Count"]["max"]),
                    value=int(st.session_state.Staff_Count), step=1, disabled=is_disabled, key="Staff_Count_num")
                st.session_state.Staff_Count = staff_count
                staff_avail = st.session_state.Staff_Availability_Percentage
                day_of_week = st.session_state.Day_of_Week
                weekend = st.session_state.Weekend
                peak_season = st.session_state.Peak_Season

            st.markdown("---")
            if st.button("Predict Energy Consumption", type="primary", use_container_width=True):
                dt = pd.to_datetime(op_date)
                input_data = {
                    "Operation_Date": [dt],
                    "Year": [dt.year],
                    "Month": [dt.month],
                    "Quarter": [dt.quarter],
                    "Day_of_Week": [day_of_week],
                    "Weekend": [weekend],
                    "Peak_Season": [peak_season],
                    "Warehouse_ID": [wh_id],
                    "Warehouse_Type": [wh_type],
                    "Warehouse_Area_sq_m": [int(wh_area)],
                    "Temperature_C": [float(temp)],
                    "Humidity_Percentage": [int(humidity)],
                    "Weather_Condition": [weather],
                    "Warehouse_Occupancy_Percentage": [int(occupancy)],
                    "Total_Shipments": [int(shipments)],
                    "Total_Cargo_Weight_kg": [float(cargo_wt)],
                    "Total_Cargo_Volume_m3": [float(cargo_vol)],
                    "Number_of_Flights": [int(n_flights)],
                    "Forklift_Operating_Hours": [float(forklift_hrs)],
                    "Conveyor_Operating_Hours": [float(conveyor_hrs)],
                    "HVAC_Operating_Hours": [float(hvac_hrs)],
                    "Lighting_Operating_Hours": [float(lighting_hrs)],
                    "Equipment_Utilization_Percentage": [float(equip_util)],
                    "Staff_Count": [int(staff_count)],
                    "Staff_Availability_Percentage": [float(staff_avail)],
                }
                input_df = pd.DataFrame(input_data)

                with st.spinner("Estimating energy consumption..."):
                    predicted_kwh = float(predictor.predict(input_df).values[0])

                col_res1, col_res2, col_res3 = st.columns(3)
                col_res1.metric("Predicted Energy Consumption", f"{predicted_kwh:.2f} kWh")
                col_res2.metric("Warehouse", f"{wh_id} ({wh_type})")
                col_res3.metric("Occupancy", f"{occupancy}%")


        # ── Tab 2: Batch CSV Upload ───────────────────────────────────────────
        with tab2:
            st.subheader("Upload Batch CSV for Predictions")
            
            template_path = "sample_batch_test.csv"
            if os.path.exists(template_path):
                try:
                    df_template = pd.read_csv(template_path)
                    if "Energy_Consumption_kWh" in df_template.columns:
                        df_template = df_template.drop(columns=["Energy_Consumption_kWh"])
                    template_csv = df_template.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download Sample Batch Template CSV",
                        data=template_csv,
                        file_name="energy_batch_template.csv",
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
                        "Operation_Date", "Year", "Month", "Quarter", "Day_of_Week", "Weekend",
                        "Peak_Season", "Warehouse_ID", "Warehouse_Type", "Warehouse_Area_sq_m",
                        "Temperature_C", "Humidity_Percentage", "Weather_Condition",
                        "Warehouse_Occupancy_Percentage", "Total_Shipments", "Total_Cargo_Weight_kg",
                        "Total_Cargo_Volume_m3", "Number_of_Flights", "Forklift_Operating_Hours",
                        "Conveyor_Operating_Hours", "HVAC_Operating_Hours", "Lighting_Operating_Hours",
                        "Equipment_Utilization_Percentage", "Staff_Count", "Staff_Availability_Percentage"
                    ]
                    missing_cols = [c for c in REQUIRED_COLS if c not in df_upload.columns]
                    if missing_cols:
                        st.error(f"Missing required columns: `{'`, `'.join(missing_cols)}`")
                        st.info("Please ensure your CSV contains all required model features. Use `sample_batch_test.csv` as a template.")
                    else:
                        if st.button("Run Batch Prediction", type="primary", use_container_width=True):
                            with st.spinner("Estimating energy consumption for all rows..."):
                                df_pred = df_upload.copy()
                                
                                num_cols = ["Warehouse_Area_sq_m", "Humidity_Percentage",
                                            "Warehouse_Occupancy_Percentage", "Total_Shipments",
                                            "Number_of_Flights", "Staff_Count"]
                                for col in num_cols:
                                    if col in df_pred.columns:
                                        df_pred[col] = pd.to_numeric(df_pred[col], errors="coerce").fillna(0).astype(int)
                                float_cols = ["Temperature_C", "Total_Cargo_Weight_kg", "Total_Cargo_Volume_m3",
                                              "Forklift_Operating_Hours", "Conveyor_Operating_Hours",
                                              "HVAC_Operating_Hours", "Lighting_Operating_Hours",
                                              "Equipment_Utilization_Percentage", "Staff_Availability_Percentage"]
                                for col in float_cols:
                                    if col in df_pred.columns:
                                        df_pred[col] = pd.to_numeric(df_pred[col], errors="coerce").astype(float)

                                for col in df_pred.select_dtypes(include=["object"]).columns:
                                    df_pred[col] = df_pred[col].astype(str).str.strip()

                                preds = predictor.predict(df_pred).values
                                df_pred["Predicted_Energy_Consumption_kWh"] = np.round(preds, 2)
                                
                                st.session_state["batch_results"] = df_pred

                    if "batch_results" in st.session_state:
                        df_res = st.session_state["batch_results"]
                        
                        total    = len(df_res)
                        avg_pred = float(df_res["Predicted_Energy_Consumption_kWh"].mean())
                        max_pred = float(df_res["Predicted_Energy_Consumption_kWh"].max())
                        min_pred = float(df_res["Predicted_Energy_Consumption_kWh"].min())

                        st.markdown("---")
                        st.markdown("### Prediction Summary")
                        s1, s2, s3, s4 = st.columns(4)
                        s1.metric("Total Records",       f"{total:,}")
                        s2.metric("Avg Predicted kWh",   f"{avg_pred:.2f}")
                        s3.metric("Max Predicted kWh",   f"{max_pred:.2f}")
                        s4.metric("Min Predicted kWh",   f"{min_pred:.2f}")

                        st.markdown("### Filter & Search Results")
                        fc1 = st.columns(1)[0]
                        with fc1:
                            search_term = st.text_input(
                                "Search by Warehouse ID",
                                placeholder="e.g. WH001",
                                key="batch_search"
                            )
                        
                        df_view = df_res.copy()
                        if search_term.strip():
                            term = search_term.strip().lower()
                            if "Warehouse_ID" in df_view.columns:
                                df_view = df_view[df_view["Warehouse_ID"].astype(str).str.lower().str.contains(term, na=False)]
                        
                        df_view = df_view.sort_values(by="Predicted_Energy_Consumption_kWh", ascending=False)
                        
                        st.markdown(f"**Showing {len(df_view):,} of {total:,} records**")
                        
                        id_cols   = [c for c in ["Warehouse_ID", "Operation_Date"] if c in df_view.columns]
                        pred_cols = ["Predicted_Energy_Consumption_kWh"]
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
                            file_name="energy_consumption_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
