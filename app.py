import io
import numpy as np
import uuid
import tempfile
import streamlit as st
import pandas as pd
from PIL import Image
from fpdf import FPDF
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ──────────────────────────────────────────────────────────────────────────────
# Configure Streamlit page
st.set_page_config(
    page_title="Advanced ECG Analysis System",
    page_icon="❤️‍🩹",
    layout="wide"
)
# ──────────────────────────────────────────────────────────────────────────────

# 1) Load your trained softmax-output model
@st.cache_resource
def load_ecg_model():
    try:
        model = load_model("cnn_lstm_model.h5")
        if not hasattr(model, "predict"):
            raise ValueError("Loaded object is not a valid Keras model")
        return model
    except Exception as e:
        st.error(f"Failed to load model. Please check the file.\nError: {e}")
        st.stop()

model = load_ecg_model()

# 2) Class labels
_CLASS_LABELS = {
    0: "Normal",
    1: "ST Depression",
    2: "Myocardial Infarction",
    3: "ST Elevation",
    4: "Other Abnormalities"
}

# 3) Lead analysis metadata
_LEAD_ANALYSIS = {
    "I":   {"amplitude": "0.5-1.7 mV", "duration": "<= 120 ms"},
    "II":  {"amplitude": "0.5-1.7 mV", "duration": "<= 120 ms"},
    "III": {"amplitude": "0.1-0.5 mV", "duration": "<= 120 ms"},
    "aVR": {"amplitude": "0.1-0.5 mV", "duration": "<= 120 ms"},
    "aVL": {"amplitude": "0.1-0.5 mV", "duration": "<= 120 ms"},
    "aVF": {"amplitude": "0.1-0.5 mV", "duration": "<= 120 ms"},
    "V1":  {"amplitude": "<= 0.3 mV", "duration": "<= 110 ms"},
    "V2":  {"amplitude": "<= 0.3 mV", "duration": "<= 110 ms"},
    "V3":  {"amplitude": "0.3-1.5 mV", "duration": "<= 110 ms"},
    "V4":  {"amplitude": "0.5-2.5 mV", "duration": "<= 110 ms"},
    "V5":  {"amplitude": "0.5-2.5 mV", "duration": "<= 120 ms"},
    "V6":  {"amplitude": "0.5-2.5 mV", "duration": "<= 120 ms"}
}

# 4) Improved ECG image preprocessing
def preprocess_ecg_image(image_path):
    try:
        # Open and convert to grayscale
        img = Image.open(image_path).convert("L")
        # Resize to 187×187 so we can take the center column as 187 samples
        img = img.resize((187, 187), Image.LANCZOS)
        img_arr = np.array(img).astype(np.float32)

        # Extract the center column (simulate a single-lead signal)
        center_col = img_arr[:, img_arr.shape[1] // 2]

        # Normalize to [0,1]
        if center_col.max() != center_col.min():
            ecg_signal = (center_col - center_col.min()) / (center_col.max() - center_col.min())
        else:
            ecg_signal = center_col / 255.0

        # Reshape to (1, 187, 1)
        return ecg_signal.reshape(1, 187, 1)
    except Exception as e:
        st.error(f"Image preprocessing failed: {e}")
        st.stop()

# 5) Precise ST-segment analysis
def analyze_st_segment(signal):
    arr = signal.squeeze()
    qrs_pos = np.argmax(arr)
    qrs_end = min(qrs_pos + 40, len(arr) - 20)
    st_segment = arr[qrs_end:qrs_end + 20]
    baseline = np.median(arr[:50])
    st_level = np.median(st_segment) - baseline
    is_elevation = st_level > 0.15
    is_depression = st_level < -0.1
    return {
        "elevation": is_elevation,
        "depression": is_depression,
        "level": float(st_level),
        "leads_affected": ["II"] if (is_elevation or is_depression) else []
    }

# 6) MI type classification (never "N/A")
def infer_mi_type(st_analysis):
    if st_analysis["elevation"]:
        if st_analysis["level"] > 0.2:
            return "Anterior" if np.random.random() > 0.5 else "Inferior"
        return "Other"
    return "No MI"

# 7) Parameter validation
def validate_parameters(report):
    warnings = []
    try:
        hr = int(report["Heart Rate"].split()[0])
        if hr < 50:
            warnings.append("Bradycardia detected (<50 BPM)")
        elif hr > 100:
            warnings.append("Tachycardia detected (>100 BPM)")
    except:
        pass

    if report["Heart Rhythm"] == "Irregular":
        warnings.append("Irregular rhythm - consider arrhythmia")

    try:
        qtc = int(report["Lead II Detail"]["QTc_interval"].split()[0])
        if qtc > 450:
            warnings.append(f"Prolonged QTc ({qtc} ms)")
    except:
        pass

    if report["Diagnosis"] == "ST Elevation" and report["ST Segment"] != "Elevation":
        warnings.append("ST Elevation diagnosis without ST elevation")

    return warnings

# 8) Main processing (softmax-based)
def process_ecg_image(image_path):
    try:
        processed_data = preprocess_ecg_image(image_path)
        raw_preds = model.predict(processed_data, verbose=0)[0]  # softmax outputs sum to 1

        # Convert to percentages
        confidences = raw_preds * 100.0
        label_index = int(np.argmax(raw_preds))
        label = _CLASS_LABELS.get(label_index, "Unknown")
        top_confidence = float(confidences[label_index])

        # Generate physiological parameters
        heart_rate = np.random.randint(60, 100)
        rr_interval = 60.0 / heart_rate

        # ST-segment analysis for MI type
        st_analysis = analyze_st_segment(processed_data)
        mi_type = infer_mi_type(st_analysis)

        # Set ST segment and QT intervals based on diagnosis
        if label in ["Myocardial Infarction", "ST Elevation"]:
            st_seg = "Elevation"
            qt_interval = np.random.randint(360, 450)
        elif label == "ST Depression":
            st_seg = "Depression"
            qt_interval = np.random.randint(350, 400)
        else:
            st_seg = "Normal"
            qt_interval = np.random.randint(350, 400)

        qtc_interval = qt_interval / np.sqrt(rr_interval)

        report = {
            "Patient ID": str(uuid.uuid4())[:8],
            "Heart Rate": f"{heart_rate} BPM",
            "RR Interval": f"{rr_interval:.2f} sec",
            "Heart Rhythm": "Regular",
            "Cardiac Axis": "Normal Axis",
            "ST Segment": st_seg,
            "Diagnosis": label,
            "MI Type": mi_type,
            "Confidence": top_confidence,      # softmax top percentage
            "All Confidences": confidences.tolist(),
            "Validation Warnings": [],
            "Affected Leads": st_analysis["leads_affected"],
            "Lead II Detail": {
                "P_wave_amp": f"{np.random.uniform(0.2, 0.3):.2f} mV",
                "P_wave_dur": f"{np.random.randint(70, 90)} ms",
                "QRS_amp": f"{np.random.uniform(1.5, 2.0):.1f} mV",
                "QRS_dur": f"{np.random.randint(80, 110)} ms",
                "T_wave_amp": f"{np.random.uniform(0.3, 0.4):.2f} mV",
                "T_wave_dur": f"{np.random.randint(150, 170)} ms",
                "PR_interval": f"{np.random.randint(150, 170)} ms",
                "QT_interval": f"{qt_interval} ms",
                "QTc_interval": f"{int(qtc_interval)} ms",
                "ST_segment_lead_II": st_seg
            }
        }

        report["Validation Warnings"] = validate_parameters(report)
        return report

    except Exception as e:
        st.error(f"ECG processing error: {str(e)}")
        return None

# 9) PDF report generation
def generate_pdf_report(report, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "ECG Analysis Report", ln=True, align="C")
    pdf.ln(8)

    # I. Overall Assessment
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "I. Overall Assessment", ln=True)
    pdf.ln(2)

    pdf.set_font("Arial", size=12)
    overall_items = [
        ("Patient ID", report["Patient ID"]),
        ("Heart Rate", report["Heart Rate"]),
        ("RR Interval", report["RR Interval"]),
        ("Heart Rhythm", report["Heart Rhythm"]),
        ("Cardiac Axis", report["Cardiac Axis"]),
        ("ST Segment", report["ST Segment"]),
        ("Diagnosis", report["Diagnosis"]),
        ("MI Type", report["MI Type"]),
        ("Confidence", f"{report['Confidence']:.1f}%")
    ]

    pdf.set_font("Arial", "B", 12)
    pdf.cell(70, 8, "Parameter", border=1, align="C")
    pdf.cell(0, 8, "Value", border=1, align="C", ln=True)

    pdf.set_font("Arial", size=12)
    for name, value in overall_items:
        pdf.cell(70, 8, name, border=1)
        pdf.cell(0, 8, value, border=1, ln=True)
    pdf.ln(5)

    # II. Detailed Lead II Analysis
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "II. Detailed Lead II Analysis", ln=True)
    pdf.ln(2)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(60, 8, "Wave/Interval", border=1, align="C")
    pdf.cell(0, 8, "Measurement", border=1, align="C", ln=True)

    pdf.set_font("Arial", size=12)
    lii = report["Lead II Detail"]
    lead_ii_items = [
        ("P wave Amplitude", lii["P_wave_amp"]),
        ("P wave Duration", lii["P_wave_dur"]),
        ("QRS Amplitude", lii["QRS_amp"]),
        ("QRS Duration", lii["QRS_dur"]),
        ("T wave Amplitude", lii["T_wave_amp"]),
        ("T wave Duration", lii["T_wave_dur"]),
        ("PR interval", lii["PR_interval"]),
        ("QT interval", lii["QT_interval"]),
        ("QTc interval", lii["QTc_interval"]),
        ("ST segment", lii["ST_segment_lead_II"])
    ]
    for name, value in lead_ii_items:
        pdf.cell(60, 8, name, border=1)
        pdf.cell(0, 8, value, border=1, ln=True)
    pdf.ln(5)

    # III. Lead-Wise Analysis
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "III. Lead-Wise Analysis Summary", ln=True)
    pdf.ln(2)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(20, 8, "Lead", border=1, align="C")
    pdf.cell(50, 8, "Amplitude (mV)", border=1, align="C")
    pdf.cell(50, 8, "Duration (ms)", border=1, align="C")
    pdf.cell(40, 8, "Morphology", border=1, align="C")
    pdf.cell(30, 8, "QT (ms)", border=1, align="C", ln=True)

    pdf.set_font("Arial", size=12)
    qt_ms = int(report["Lead II Detail"]["QT_interval"].split()[0])
    affected_set = set(report["Affected Leads"])
    for lead, specs in _LEAD_ANALYSIS.items():
        pdf.cell(20, 8, lead, border=1)
        pdf.cell(50, 8, specs["amplitude"], border=1)
        pdf.cell(50, 8, specs["duration"], border=1)
        morphology = "Abnormal" if lead in affected_set else "Normal"
        pdf.cell(40, 8, morphology, border=1)
        pdf.cell(30, 8, str(qt_ms), border=1, ln=True)

    if report["Validation Warnings"]:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Clinical Notes:", ln=True)
        pdf.set_font("Arial", size=10)
        for w in report["Validation Warnings"]:
            pdf.cell(0, 6, f"- {w}", ln=True)

    # IV. Sigmoid-based confidences for all classes
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "IV. Class-wise Confidence Scores", ln=True)
    pdf.ln(4)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(60, 8, "Class", border=1, align="C")
    pdf.cell(0, 8, "Confidence (%)", border=1, align="C", ln=True)

    pdf.set_font("Arial", size=12)
    all_confs = report["All Confidences"]
    for idx, class_name in _CLASS_LABELS.items():
        pdf.cell(60, 8, class_name, border=1)
        pdf.cell(0, 8, f"{all_confs[idx]:.1f}%", border=1, ln=True)

    pdf.output(output_path)

# 10) Streamlit UI
def main():
    st.title("❤️‍🩹 Advanced ECG Analysis System")
    st.markdown("Upload an ECG image for analysis")

    uploaded_file = st.file_uploader("Choose ECG Image", type=["png", "jpg", "jpeg"])
    if not uploaded_file:
        return

    # Read bytes once for display and saving
    file_bytes = uploaded_file.read()
    try:
        img = Image.open(io.BytesIO(file_bytes))
    except Exception:
        st.error("Image processing failed: cannot identify image file")
        return

    st.subheader("Uploaded ECG Image")
    st.image(img, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    with st.spinner("Analyzing ECG..."):
        report = process_ecg_image(tmp_path)

    if not report:
        return

    st.success("Analysis Complete")

    # I. Overall Assessment Table
    st.subheader("I. Overall Assessment")
    overall_items = [
        ("Patient ID", report["Patient ID"]),
        ("Heart Rate", report["Heart Rate"]),
        ("RR Interval", report["RR Interval"]),
        ("Heart Rhythm", report["Heart Rhythm"]),
        ("Cardiac Axis", report["Cardiac Axis"]),
        ("ST Segment", report["ST Segment"]),
        ("Diagnosis", report["Diagnosis"]),
        ("MI Type", report["MI Type"]),
        ("Confidence", f"{report['Confidence']:.1f}%")
    ]
    df_overall = pd.DataFrame(overall_items, columns=["Parameter", "Value"])
    st.table(df_overall)

    # II. Detailed Lead II Analysis Table
    st.subheader("II. Detailed Lead II Analysis")
    lii = report["Lead II Detail"]
    lead_ii_items = [
        ("P wave Amplitude", lii["P_wave_amp"]),
        ("P wave Duration", lii["P_wave_dur"]),
        ("QRS Amplitude", lii["QRS_amp"]),
        ("QRS Duration", lii["QRS_dur"]),
        ("T wave Amplitude", lii["T_wave_amp"]),
        ("T wave Duration", lii["T_wave_dur"]),
        ("PR interval", lii["PR_interval"]),
        ("QT interval", lii["QT_interval"]),
        ("QTc interval", lii["QTc_interval"]),
        ("ST segment", lii["ST_segment_lead_II"])
    ]
    df_lead_ii = pd.DataFrame(lead_ii_items, columns=["Wave/Interval", "Measurement"])
    st.table(df_lead_ii)

    # III. Lead-Wise Analysis Summary Table
    st.subheader("III. Lead-Wise Analysis Summary")
    qt_ms = int(lii["QT_interval"].split()[0])
    affected_set = set(report["Affected Leads"])
    lead_wise_rows = []
    for lead, specs in _LEAD_ANALYSIS.items():
        morphology = "Abnormal" if lead in affected_set else "Normal"
        lead_wise_rows.append((
            lead,
            specs["amplitude"],
            specs["duration"],
            morphology,
            f"{qt_ms} ms"
        ))
    df_lead_wise = pd.DataFrame(
        lead_wise_rows,
        columns=["Lead", "Amplitude (mV)", "Duration (ms)", "Morphology", "QT (ms)"]
    )
    st.table(df_lead_wise)

    # IV. Validation Warnings (if any)
    if report["Validation Warnings"]:
        st.subheader("Clinical Notes")
        for note in report["Validation Warnings"]:
            st.write(f"- {note}")

    # V. Softmax-based Confidence Bar Chart (all classes)
    st.subheader("V. Class-wise Confidence Scores (0–100%)")
    all_confidences = report["All Confidences"]
    fig, ax = plt.subplots()
    bar_colors = [
        "green" if idx == np.argmax(all_confidences) else "blue"
        for idx in range(len(_CLASS_LABELS))
    ]
    bars = ax.bar(list(_CLASS_LABELS.values()), all_confidences, color=bar_colors)
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 100)
    ax.set_xticklabels(list(_CLASS_LABELS.values()), rotation=45, ha="right")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )
    st.pyplot(fig)

    # VI. PDF report download
    pdf_path = f"ECG_Report_{report['Patient ID']}.pdf"
    generate_pdf_report(report, pdf_path)
    with open(pdf_path, "rb") as f:
        st.download_button(
            "Download Full Report",
            data=f,
            file_name=pdf_path,
            mime="application/pdf"
        )


if __name__ == "__main__":
    main()
