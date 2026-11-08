
from pathlib import Path
import html

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CLAIRITY - INTELLIGENT AIRWAY MONITORING
# BioHack Research Prototype
#
# Recorded ventilator features + simulated sensor measurements
# NOT FOR CLINICAL USE
# ============================================================


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CLAIRITY | Airway Monitoring",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PATHS AND COLORS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PLOTS_DIR = BASE_DIR / "plots"

DATA_FILE = PROCESSED_DIR / "clairity_hybrid_v2.csv"
MODEL_FILE = DATA_DIR / "clairity_model_v2.pkl"

COMPARISON_FILE = PROCESSED_DIR / "model_comparison_v2.csv"
PREDICTIONS_FILE = (
    PROCESSED_DIR / "clairity_hybrid_predictions_v2.csv"
)

BLUE = "#0066CC"
PURPLE = "#7E22CE"
GREEN = "#059669"
ORANGE = "#D97706"
RED = "#DC2626"
DARK = "#0F172A"
GRAY = "#64748B"
LIGHT = "#F8FAFC"
BORDER = "#E2E8F0"

STATUS_COLORS = {
    "normal": GREEN,
    "at_risk": ORANGE,
    "critical": RED,
    "sensor_failure": PURPLE,
}

STATUS_NAMES = {
    "normal": "NORMAL",
    "at_risk": "AT RISK",
    "critical": "CRITICAL",
    "sensor_failure": "SENSOR FAILURE",
}

SENSOR_COLUMNS = {
    "capacitance_above": "Capacitance Above Cuff",
    "capacitance_below": "Capacitance Below Cuff",
    "ph_above": "pH Above Cuff",
    "ph_below": "pH Below Cuff",
}

CLASS_ORDER = [
    "normal",
    "at_risk",
    "critical",
    "sensor_failure",
]


# ============================================================
# 3. MINIMALIST CLAIRITY THEME
# ============================================================

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap'
    );

    :root {
        color-scheme: light;
    }

    html, body, .stApp {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    [data-testid="stSidebar"] * {
        color: #0F172A;
    }

    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        color: #0F172A;
        letter-spacing: -0.035em;
    }

    h2 {
        font-weight: 750;
    }

    p, label {
        color: #475569;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 13px;
        padding: 18px;
        min-height: 115px;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B;
    }

    div[data-testid="stMetricValue"] {
        color: #0F172A;
        font-family: 'JetBrains Mono', monospace;
        font-size: 27px;
    }

    .clairity-header {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 25px 30px;
        margin-bottom: 24px;
    }

    .clairity-logo {
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -2px;
        line-height: 1.2;
    }

    .clairity-subtitle {
        color: #64748B;
        font-size: 14px;
        margin-top: 7px;
    }

    .research-label {
        color: #D97706;
        font-size: 11px;
        font-weight: 700;
        margin-top: 13px;
        letter-spacing: 0.035em;
    }

    .status-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid var(--status-color);
        border-radius: 14px;
        padding: 23px;
        margin: 14px 0 24px;
    }

    .status-heading {
        color: #64748B;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
    }

    .status-value {
        color: var(--status-color);
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }

    .status-description {
        color: #64748B;
        font-size: 12px;
        margin-top: 8px;
    }

    .section-heading {
        color: #475569;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.09em;
        margin: 24px 0 12px;
    }

    div[data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 7px;
    }

    div[data-testid="stButton"] button {
        border-radius: 9px;
        font-weight: 600;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    .small-note {
        color: #64748B;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. LOAD DATA AND MODEL
# ============================================================

@st.cache_data(show_spinner="Loading CLAIRITY dataset...")
def load_dataset():
    return pd.read_csv(DATA_FILE)


@st.cache_resource(show_spinner="Loading CLAIRITY model...")
def load_model():
    return joblib.load(MODEL_FILE)


if not DATA_FILE.exists():
    st.error(f"Missing dataset: {DATA_FILE}")
    st.stop()

if not MODEL_FILE.exists():
    st.error(f"Missing trained model: {MODEL_FILE}")
    st.stop()

try:
    df = load_dataset()
    saved_model = load_model()
except Exception as exc:
    st.error(f"Unable to load project files: {exc}")
    st.stop()

if "condition" not in df.columns:
    st.error("The dataset must contain a 'condition' column.")
    st.stop()

if "recording_id" not in df.columns:
    st.error("The dataset must contain a 'recording_id' column.")
    st.stop()

# Support a model saved directly or inside a dictionary.
if isinstance(saved_model, dict):
    model = saved_model.get("model")
    model_features = saved_model.get("features")
else:
    model = saved_model
    model_features = None

if model is None:
    st.error("The saved model package does not contain a model.")
    st.stop()

if model_features is None:
    model_features = getattr(
        model,
        "feature_names_in_",
        None,
    )

if model_features is None:
    st.error(
        "The saved model does not specify its input features. "
        "Please save the training feature list with the model."
    )
    st.stop()

model_features = list(model_features)

missing_features = [
    feature
    for feature in model_features
    if feature not in df.columns
]

if missing_features:
    st.error(
        "These model features are missing from the dataset: "
        + ", ".join(missing_features)
    )
    st.stop()


# ============================================================
# 5. GENERATE PREDICTIONS
# ============================================================

@st.cache_data(show_spinner="Analyzing sensor data...")
def generate_predictions(_model, dataset, features):
    X = dataset[features].copy()
    return _model.predict(X)


try:
    df["prediction"] = generate_predictions(
        model,
        df,
        model_features,
    )
except Exception as exc:
    st.error(f"Model prediction failed: {exc}")
    st.info(
        "Check that the saved model uses the same preprocessing "
        "and features as the CLAIRITY V2 dataset."
    )
    st.stop()

df["correct_prediction"] = (
    df["condition"] == df["prediction"]
)


# ============================================================
# 6. HELPER FUNCTIONS
# ============================================================

def safe_number(value, decimals=2):
    if pd.isna(value):
        return "N/A"

    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def section_title(text):
    st.markdown(
        f'<div class="section-heading">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def display_status(condition):
    condition = str(condition)

    color = STATUS_COLORS.get(condition, GRAY)
    label = STATUS_NAMES.get(
        condition,
        condition.upper(),
    )

    st.markdown(
        f"""
        <div class="status-card"
             style="--status-color:{color};">
            <div class="status-heading">
                CURRENT AI PREDICTION
            </div>
            <div class="status-value">
                {html.escape(label)}
            </div>
            <div class="status-description">
                Experimental classification based on
                recorded ventilator features and
                simulated sensor readings
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_line_chart(data, column, title, color):
    fig = go.Figure()

    if column in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data["_display_index"],
                y=data[column],
                mode="lines+markers",
                line=dict(
                    color=color,
                    width=2.5,
                ),
                marker=dict(size=5),
                name=title,
                connectgaps=False,
                hovertemplate=(
                    "Record: %{x}<br>"
                    "Value: %{y:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=15, color=DARK),
        ),
        template="plotly_white",
        height=300,
        margin=dict(l=30, r=20, t=55, b=40),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Inter",
            color=GRAY,
        ),
        xaxis_title="Breath record",
        yaxis_title="Value",
        hovermode="x unified",
        showlegend=False,
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#F1F5F9",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#F1F5F9",
    )

    return fig


def show_alert(condition):
    if condition == "normal":
        st.success(
            "NORMAL: No concerning pattern detected "
            "in this simulated case."
        )

    elif condition == "at_risk":
        st.warning(
            "AT RISK: The model detected a simulated "
            "pattern associated with possible fluid accumulation."
        )

    elif condition == "critical":
        st.error(
            "CRITICAL: The model detected a simulated "
            "pattern associated with possible fluid movement "
            "below the cuff."
        )

    elif condition == "sensor_failure":
        st.warning(
            "SENSOR FAILURE: One or more simulated sensor "
            "readings may be unavailable or unreliable."
        )

    else:
        st.info(f"Model output: {condition}")

    st.caption(
        "These messages are demonstration outputs, "
        "not clinical alarms or medical recommendations."
    )


def plotly_display(fig):
    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displaylogo": False,
            "scrollZoom": False,
        },
    )


def show_metric(row, column, title, decimals=2):
    if column in row.index:
        st.metric(
            title,
            safe_number(row[column], decimals),
        )
    else:
        st.metric(title, "N/A")


# ============================================================
# 7. DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <div class="clairity-header">
        <div class="clairity-logo">
            <span style="color:#0F172A;">CL</span><span style="color:#0066CC;">AIR</span><span style="color:#0F172A;">ITY</span>
        </div>
        <div class="clairity-subtitle">
            Intelligent Airway Monitoring & Diagnostic Research
        </div>
        <div class="research-label">
            RESEARCH PROTOTYPE · SIMULATED SENSOR DATA ·
            NOT FOR CLINICAL USE
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 8. SIDEBAR
# ============================================================

st.sidebar.markdown("## CLAIRITY Controls")

page = st.sidebar.radio(
    "Navigation",
    [
        "Live Monitoring",
        "Model Performance",
        "Sensor Analysis",
        "Prediction Errors",
    ],
)

st.sidebar.divider()

recording_ids = (
    df["recording_id"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

recording_ids = sorted(recording_ids)

selected_recording = st.sidebar.selectbox(
    "Select ventilator recording",
    recording_ids,
)

recording_data = df[
    df["recording_id"].astype(str)
    == selected_recording
].copy()

# Keep breaths in their original sequence.
if "breath_id" in recording_data.columns:
    recording_data = recording_data.sort_values(
        "breath_id",
        kind="stable",
    )

recording_data = recording_data.reset_index(drop=True)

recording_data["_display_index"] = np.arange(
    len(recording_data)
)

st.sidebar.caption(
    f"{len(recording_data):,} breath-level records "
    "available for this recording."
)

st.sidebar.warning(
    "Replay mode uses existing data. "
    "No physical medical device is connected."
)

st.sidebar.divider()

st.sidebar.caption(
    "CLAIRITY | BioHack Research Prototype"
)


# ============================================================
# 9. LIVE MONITORING
# ============================================================

if page == "Live Monitoring":

    st.header("Live Monitoring")

    st.caption(
        "Live-style replay of recorded ventilator features "
        "and simulated sensor measurements."
    )

    # Session state for replay
    if "replay_recording" not in st.session_state:
        st.session_state.replay_recording = selected_recording

    if (
        st.session_state.replay_recording
        != selected_recording
    ):
        st.session_state.replay_recording = selected_recording
        st.session_state.replay_position = 0
        st.session_state.replay_running = False

    if "replay_position" not in st.session_state:
        st.session_state.replay_position = 0

    if "replay_running" not in st.session_state:
        st.session_state.replay_running = False

    # Controls
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        if st.button(
            "▶ Start / Resume",
            width="stretch",
            type="primary",
        ):
            if (
                st.session_state.replay_position
                >= len(recording_data) - 1
            ):
                st.session_state.replay_position = 0

            st.session_state.replay_running = True

    with c2:
        if st.button(
            "⏸ Pause",
            width="stretch",
        ):
            st.session_state.replay_running = False

    with c3:
        if st.button(
            "↺ Restart",
            width="stretch",
        ):
            st.session_state.replay_position = 0
            st.session_state.replay_running = False

    history_length = st.slider(
        "Number of recent records to display",
        min_value=10,
        max_value=100,
        value=40,
        step=10,
    )

    if not st.session_state.replay_running:
        selected_position = st.slider(
            "Manually select a breath record",
            min_value=0,
            max_value=max(0, len(recording_data) - 1),
            value=min(
                st.session_state.replay_position,
                len(recording_data) - 1,
            ),
        )

        st.session_state.replay_position = (
            selected_position
        )

    st.caption(
        "Replay advances approximately once per second. "
        "Use Pause to inspect individual readings."
    )

    # Fragment refreshes only the monitoring section.
    # No blocking time.sleep() loop.
    @st.fragment(run_every="1s")
    def live_monitor():

        if st.session_state.replay_running:
            if (
                st.session_state.replay_position
                < len(recording_data) - 1
            ):
                st.session_state.replay_position += 1
            else:
                st.session_state.replay_running = False

        position = st.session_state.replay_position

        latest = recording_data.iloc[position]

        progress = (
            (position + 1) / len(recording_data)
        )

        st.progress(
            progress,
            text=(
                f"Breath record {position + 1:,} "
                f"of {len(recording_data):,}"
            ),
        )

        display_status(latest["prediction"])

        section_title("SENSOR READINGS")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            show_metric(
                latest,
                "capacitance_above",
                "Capacitance Above",
            )

        with m2:
            show_metric(
                latest,
                "capacitance_below",
                "Capacitance Below",
            )

        with m3:
            show_metric(
                latest,
                "ph_above",
                "pH Above Cuff",
            )

        with m4:
            show_metric(
                latest,
                "ph_below",
                "pH Below Cuff",
            )

        section_title("VENTILATOR FEATURES")

        v1, v2, v3 = st.columns(3)

        with v1:
            show_metric(
                latest,
                "signal_1_mean",
                "Signal 1 Mean",
            )

        with v2:
            show_metric(
                latest,
                "signal_2_mean",
                "Signal 2 Mean",
            )

        with v3:
            show_metric(
                latest,
                "sample_count",
                "Samples per Breath",
                decimals=0,
            )

        # Build a rolling window.
        start = max(
            0,
            position - history_length + 1,
        )

        history = recording_data.iloc[
            start:position + 1
        ].copy()

        section_title("LIVE SENSOR TRENDS")

        left, right = st.columns(2)

        with left:
            plotly_display(
                make_line_chart(
                    history,
                    "capacitance_above",
                    "Capacitance Above Cuff",
                    BLUE,
                )
            )

            plotly_display(
                make_line_chart(
                    history,
                    "ph_above",
                    "pH Above Cuff",
                    PURPLE,
                )
            )

        with right:
            plotly_display(
                make_line_chart(
                    history,
                    "capacitance_below",
                    "Capacitance Below Cuff",
                    BLUE,
                )
            )

            plotly_display(
                make_line_chart(
                    history,
                    "ph_below",
                    "pH Below Cuff",
                    PURPLE,
                )
            )

        section_title("VENTILATOR SIGNAL TRENDS")

        vent_left, vent_right = st.columns(2)

        with vent_left:
            plotly_display(
                make_line_chart(
                    history,
                    "signal_1_mean",
                    "Signal 1 Mean by Breath",
                    BLUE,
                )
            )

        with vent_right:
            plotly_display(
                make_line_chart(
                    history,
                    "signal_2_mean",
                    "Signal 2 Mean by Breath",
                    PURPLE,
                )
            )

        st.caption(
            "These plots show breath-level signal "
            "summaries, not raw ventilator waveforms. "
            "The signals have not been assigned "
            "confirmed physiological identities."
        )

        section_title("AUTOMATED SIMULATED ALERTS")

        show_alert(latest["prediction"])

        section_title("RECENT DETECTION HISTORY")

        history_columns = [
            "breath_id",
            "condition",
            "prediction",
            "capacitance_above",
            "capacitance_below",
            "ph_above",
            "ph_below",
        ]

        history_columns = [
            column
            for column in history_columns
            if column in history.columns
        ]

        st.dataframe(
            history[history_columns].tail(15),
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "'Condition' is the synthetic reference label. "
            "'Prediction' is the machine-learning output."
        )

        if (
            position == len(recording_data) - 1
            and not st.session_state.replay_running
        ):
            st.info(
                "End of recording. Press Restart "
                "to replay it."
            )

    live_monitor()


# ============================================================
# 10. MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("Machine Learning Performance")

    st.info(
        "Performance figures describe a synthetic-label "
        "research experiment. They do not measure "
        "clinical diagnostic accuracy."
    )

    if COMPARISON_FILE.exists():

        comparison = pd.read_csv(COMPARISON_FILE)

        st.subheader("Model Comparison")

        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True,
        )

        if {
            "model",
            "accuracy",
        }.issubset(comparison.columns):

            plot_data = comparison.copy()

            plot_data["accuracy_pct"] = (
                plot_data["accuracy"] * 100
            )

            fig = px.bar(
                plot_data,
                x="model",
                y="accuracy_pct",
                text="accuracy_pct",
                title="Model Accuracy Comparison",
                color="model",
                color_discrete_sequence=[
                    GRAY,
                    PURPLE,
                    BLUE,
                ],
            )

            fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
            )

            fig.update_layout(
                template="plotly_white",
                showlegend=False,
                yaxis_title="Accuracy (%)",
                xaxis_title="Model",
                yaxis_range=[0, 105],
                height=440,
            )

            plotly_display(fig)

    else:
        st.warning(
            "Model comparison CSV not found."
        )

    st.subheader("Held-Out Test Predictions")

    if PREDICTIONS_FILE.exists():

        test_df = pd.read_csv(PREDICTIONS_FILE)

        actual_col = "condition"
        predicted_col = "predicted_condition"

        if {
            actual_col,
            predicted_col,
        }.issubset(test_df.columns):

            actual = test_df[actual_col]
            predicted = test_df[predicted_col]

            accuracy = accuracy_score(
                actual,
                predicted,
            )

            errors = (actual != predicted).sum()

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Test Accuracy",
                f"{accuracy * 100:.2f}%",
            )

            m2.metric(
                "Test Cases",
                f"{len(test_df):,}",
            )

            m3.metric(
                "Incorrect Predictions",
                f"{errors:,}",
            )

            labels = [
                label
                for label in CLASS_ORDER
                if (
                    label in actual.values
                    or label in predicted.values
                )
            ]

            matrix = confusion_matrix(
                actual,
                predicted,
                labels=labels,
            )

            fig = go.Figure(
                data=go.Heatmap(
                    z=matrix,
                    x=labels,
                    y=labels,
                    colorscale="Blues",
                    text=matrix,
                    texttemplate="%{text}",
                    hovertemplate=(
                        "Actual: %{y}<br>"
                        "Predicted: %{x}<br>"
                        "Cases: %{z}"
                        "<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                title="Hybrid Model Confusion Matrix",
                xaxis_title="Predicted Condition",
                yaxis_title="Actual Condition",
                template="plotly_white",
                height=500,
            )

            plotly_display(fig)

            report = classification_report(
                actual,
                predicted,
                labels=labels,
                output_dict=True,
                zero_division=0,
            )

            report_df = (
                pd.DataFrame(report)
                .transpose()
            )

            st.subheader("Classification Report")

            st.dataframe(
                report_df.round(3),
                width="stretch",
            )

            st.warning(
                "Important limitation: A model may "
                "misclassify a simulated critical case "
                "as normal or at risk. Sensor-failure "
                "detection may also be unusually easy "
                "because missing sensor values were "
                "introduced systematically."
            )

        else:
            st.warning(
                "The prediction CSV has different "
                "column names than expected."
            )

    else:
        st.warning(
            "Held-out prediction CSV not found."
        )


# ============================================================
# 11. SENSOR ANALYSIS
# ============================================================

elif page == "Sensor Analysis":

    st.header("Sensor Analysis")

    st.caption(
        "Explore the distributions of simulated "
        "capacitance and fluorescence-based pH "
        "sensor readings."
    )

    available_sensors = [
        sensor
        for sensor in SENSOR_COLUMNS
        if sensor in df.columns
    ]

    sensor = st.selectbox(
        "Select sensor",
        available_sensors,
        format_func=lambda x: SENSOR_COLUMNS[x],
    )

    fig = go.Figure()

    for condition in CLASS_ORDER:

        values = df.loc[
            df["condition"] == condition,
            sensor,
        ].dropna()

        if values.empty:
            continue

        fig.add_trace(
            go.Histogram(
                x=values,
                name=STATUS_NAMES[condition],
                marker_color=STATUS_COLORS[condition],
                opacity=0.55,
                histnorm="probability density",
                nbinsx=50,
            )
        )

    fig.update_layout(
        title=SENSOR_COLUMNS[sensor],
        barmode="overlay",
        template="plotly_white",
        height=500,
        xaxis_title="Simulated sensor value",
        yaxis_title="Probability density",
        legend_title="Condition",
    )

    plotly_display(fig)

    st.subheader("Statistical Summary")

    summary = (
        df.groupby("condition")[sensor]
        .describe()
        .round(3)
    )

    st.dataframe(
        summary,
        width="stretch",
    )

    st.subheader("Missing Sensor Values")

    missing = (
        df[list(SENSOR_COLUMNS.keys())]
        .isna()
        .sum()
        .reset_index()
    )

    missing.columns = [
        "Sensor",
        "Missing Values",
    ]

    st.dataframe(
        missing,
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Overlapping distributions mean different "
        "simulated conditions can produce similar "
        "readings. Missing below-cuff values may "
        "represent the simulated sensor-failure class."
    )


# ============================================================
# 12. PREDICTION ERROR ANALYSIS
# ============================================================

elif page == "Prediction Errors":

    st.header("Prediction Error Analysis")

    if not PREDICTIONS_FILE.exists():
        st.warning(
            "Prediction results file not found."
        )
        st.stop()

    test_df = pd.read_csv(PREDICTIONS_FILE)

    required = [
        "condition",
        "predicted_condition",
    ]

    if not all(
        column in test_df.columns
        for column in required
    ):
        st.error(
            "Prediction file is missing the expected "
            "condition columns."
        )
        st.stop()

    errors = test_df[
        test_df["condition"]
        != test_df["predicted_condition"]
    ].copy()

    total_errors = len(errors)
    total_cases = len(test_df)

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Total Test Cases",
        f"{total_cases:,}",
    )

    c2.metric(
        "Incorrect Predictions",
        f"{total_errors:,}",
    )

    c3.metric(
        "Error Rate",
        (
            f"{total_errors / total_cases * 100:.2f}%"
            if total_cases
            else "N/A"
        ),
    )

    st.subheader("Where the Model Makes Mistakes")

    if not errors.empty:

        counts = (
            errors.groupby(
                [
                    "condition",
                    "predicted_condition",
                ]
            )
            .size()
            .reset_index(name="count")
        )

        fig = px.bar(
            counts,
            x="condition",
            y="count",
            color="predicted_condition",
            barmode="group",
            title="Prediction Errors by Condition",
            labels={
                "condition": "Actual Condition",
                "predicted_condition": "Predicted Condition",
                "count": "Number of Errors",
            },
            color_discrete_map=STATUS_COLORS,
        )

        fig.update_layout(
            template="plotly_white",
            height=460,
        )

        plotly_display(fig)

        st.subheader("Explore Incorrect Predictions")

        filter_condition = st.selectbox(
            "Filter by actual condition",
            ["All"] + sorted(
                errors["condition"].unique().tolist()
            ),
        )

        filtered = errors.copy()

        if filter_condition != "All":
            filtered = filtered[
                filtered["condition"]
                == filter_condition
            ]

        display_columns = [
            "recording_id",
            "breath_id",
            "condition",
            "predicted_condition",
            "capacitance_above",
            "capacitance_below",
            "ph_above",
            "ph_below",
        ]

        display_columns = [
            column
            for column in display_columns
            if column in filtered.columns
        ]

        st.dataframe(
            filtered[display_columns],
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download Prediction Errors",
            data=filtered.to_csv(index=False),
            file_name="clairity_prediction_errors.csv",
            mime="text/csv",
        )

    else:
        st.success(
            "No errors found in the provided "
            "prediction file."
        )

    st.warning(
        "These errors are evaluated against "
        "synthetic condition labels, not verified "
        "clinical outcomes."
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "CLAIRITY | BioHack 2026 | "
    "Recorded ventilator data + simulated sensors | "
    "Research demonstration only | "
    "Not for clinical use"
)
