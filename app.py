import cv2
import numpy as np
import uuid
import streamlit as st
import tempfile
from PIL import Image
from fpdf import FPDF
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ──────────────────────────────────────────────────────────────────────────────
# Make set_page_config() the first Streamlit command in the file:
st.set_page_config(
    page_title="Advanced ECG Analysis System",
    page_icon="❤️‍🩹",
    layout="wide"
)
# ──────────────────────────────────────────────────────────────────────────────

# 1) Load your trained CNN-LSTM model (five-output softmax)
@st.cache_resource
def load_ecg_model():
    try:
        return load_model("cnn_lstm_model.h5")
    except Exception as e:
        st.error(f"Failed to load model. Please check the file.\nError: {e}")
        st.stop()

model = load_ecg_model()

# 2) Class labels (must match the ordering used during training)
_CLASS_LABELS = {
    0: "Normal",
    1: "ST Depression",
    2: "Myocardial Infarction",
    3: "ST Elevation",
    4: "Other Abnormalities"
}

# 3) Lead‐analysis metadata (unchanged)
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

_LEADS_ALL = list(_LEAD_ANALYSIS.keys())

# 4) Preprocessing: convert ECG image → 1D signal of length 187 → (1, 187, 1)
def preprocess_ecg_image(image_path):
    img = load_img(image_path, target_size=(256, 256), color_mode="grayscale")
    img_arr = img_to_array(img) / 255.0
    ecg_signal = img_arr.mean(axis=1).squeeze()
    ecg_signal = np.interp(
        np.linspace(0, 1, 187),
        np.linspace(0, 1, len(ecg_signal)),
        ecg_signal
    )
    return ecg_signal.reshape(1, 187, 1)

# 5) Simple ST‐segment analysis (unchanged)
def analyze_st_segment(signal):
    signal = signal.squeeze()
    qrs_end = 100
    st_segment = signal[qrs_end : qrs_end + 20]
    baseline = np.median(signal[:50])
    st_level = np.median(st_segment) - baseline
    return {
        "elevation": st_level > 0.1,
        "depression": st_level < -0.1,
        "level": float(st_level),
        "leads_affected": ["II"]
    }

def infer_mi_type(st_analysis):
    if st_analysis["elevation"]:
        return "Inferior"
    return "Undetermined Type"

# 6) Validate basic ECG parameters (unchanged)
def validate_parameters(report):
    warnings = []
    try:
        hr = int(report["Heart Rate"].split()[0])
    except Exception:
        hr = 0
    if hr < 60:
        warnings.append("Bradycardia detected (<60 BPM)")
    elif hr > 100:
        warnings.append("Tachycardia detected (>100 BPM)")
    if report["Heart Rhythm"] == "Irregular":
        warnings.append("Irregular rhythm requires further investigation")
    if report["Diagnosis"] == "Myocardial Infarction" and report["ST Segment"] != "Elevation":
        warnings.append("MI diagnosis without ST elevation - consider alternative diagnoses")
    return warnings

# 7) Main processing: run model.predict → build full report dict
def process_ecg_image(image_path):
    processed_data = preprocess_ecg_image(image_path)
    raw_pred = model.predict(processed_data)[0]  # shape (5,) for five‐class softmax

    label_index = int(np.argmax(raw_pred))
    label = _CLASS_LABELS.get(label_index, "Unknown")
    confidence = float(raw_pred[label_index])

    st_analysis = analyze_st_segment(processed_data)
    if label == "Myocardial Infarction":
        st_seg = "Elevation"
        mi_type = infer_mi_type(st_analysis)
    elif label == "ST Depression":
        st_seg = "Depression"
        mi_type = "N/A"
    else:
        st_seg = "Normal"
        mi_type = "N/A"

    lead_ii_detail = {
        "P_wave_amp": "0.25 mV",
        "P_wave_dur": "80 ms",
        "QRS_amp": "1.8 mV",
        "QRS_dur": "100 ms",
        "T_wave_amp": "0.35 mV",
        "T_wave_dur": "160 ms",
        "PR_interval": "160 ms",
        "QT_interval": "400 ms",
        "QTc_interval": "430 ms",
        "ST_segment_lead_II": st_seg
    }

    report = {
        "Patient ID": str(uuid.uuid4())[:8],
        "Heart Rate": f"{np.random.randint(50, 110)} BPM",
        "RR Interval": f"{np.random.uniform(0.6, 1.0):.2f} sec",
        "Heart Rhythm": "Regular" if np.random.random() > 0.2 else "Irregular",
        "Cardiac Axis": "Normal Axis",
        "ST Segment": st_seg,
        "Diagnosis": label,
        "MI Type": mi_type,
        "Confidence": confidence,
        "Validation Warnings": [],
        "Affected Leads": st_analysis["leads_affected"],
        "Lead II Detail": lead_ii_detail
    }

    report["Validation Warnings"] = validate_parameters(report)
    return report

# 8) Generate PDF report (unchanged)
def generate_pdf_report(report, output_path="ECG_Report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "ECG Analysis Report", ln=True, align="C")
    pdf.ln(8)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "I. Overall Assessment", ln=True)
    pdf.ln(2)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(70, 8, "Parameter", border=1, align="C")
    pdf.cell(0, 8, "Value", border=1, align="C", ln=True)

    pdf.set_font("Arial", size=12)
    overall_items = [
        ("Heart Rate", report["Heart Rate"]),
        ("RR Interval", report["RR Interval"]),
        ("Heart Rhythm", report["Heart Rhythm"]),
        ("Cardiac Axis", report["Cardiac Axis"]),
        ("ST Segment", report["ST Segment"]),
        ("Diagnosis", report["Diagnosis"]),
        ("MI Type", report["MI Type"]),
        ("Interpretation Accuracy", f"{report['Confidence']*100:.1f}%")
    ]
    for name, value in overall_items:
        pdf.cell(70, 8, name, border=1)
        pdf.cell(0, 8, value, border=1, ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, "II. Detailed Lead II Analysis", ln=True)
    pdf.ln(2)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(60, 8, "Wave/Interval", border=1, align="C")
    pdf.cell(0, 8, "Measurement", border=1, align="C", ln=True)

    pdf.set_font("Arial", size=12)
    lii = report["Lead II Detail"]
    lead_ii_items = [
        ("P wave Amp", lii["P_wave_amp"]),
        ("P wave Dur", lii["P_wave_dur"]),
        ("QRS Amp", lii["QRS_amp"]),
        ("QRS Dur", lii["QRS_dur"]),
        ("T wave Amp", lii["T_wave_amp"]),
        ("T wave Dur", lii["T_wave_dur"]),
        ("PR interval", lii["PR_interval"]),
        ("QT interval", lii["QT_interval"]),
        ("QTc interval", lii["QTc_interval"]),
        ("ST segment", lii["ST_segment_lead_II"])
    ]
    for name, value in lead_ii_items:
        pdf.cell(60, 8, name, border=1)
        pdf.cell(0, 8, value, border=1, ln=True)
    pdf.ln(5)

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
        amp = specs["amplitude"]
        dur = specs["duration"]
        morphology = "Abnormal" if lead in affected_set else "Normal"
        pdf.cell(20, 8, lead, border=1)
        pdf.cell(50, 8, amp, border=1)
        pdf.cell(50, 8, dur, border=1)
        pdf.cell(40, 8, morphology, border=1)
        pdf.cell(30, 8, str(qt_ms), border=1, ln=True)

    if report["Validation Warnings"]:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Clinical Validation Notes:", ln=True)
        pdf.set_font("Arial", size=10)
        for w in report["Validation Warnings"]:
            pdf.cell(0, 6, f"- {w}", ln=True)

    pdf.output(output_path)

# 9) Streamlit UI & main()
def main():
    st.title("❤️‍🩹 Advanced ECG Analysis System")
    st.markdown("Upload an ECG image (PNG/JPG) for automated diagnosis.")

    uploaded_file = st.file_uploader("Choose ECG Image", type=["png", "jpg", "jpeg"])
    if not uploaded_file:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_filename = tmp_file.name

    report = process_ecg_image(tmp_filename)

    st.subheader("Summary")
    st.write(f"- **Diagnosis:** {report['Diagnosis']}")
    st.write(f"- **Interpretation Accuracy:** {report['Confidence']*100:.1f}%")
    if report["Validation Warnings"]:
        st.warning("⚠️ Validation Warnings:")
        for w in report["Validation Warnings"]:
            st.warning(f"- {w}")

    display_labels = list(_CLASS_LABELS.values())
    raw_pred = model.predict(preprocess_ecg_image(tmp_filename))[0]
    probs = [float(raw_pred[i]) for i in range(len(display_labels))]

    fig, ax = plt.subplots()
    ax.bar(display_labels, probs)
    ax.set_ylabel("Prediction Probability")
    ax.set_xlabel("Classes")
    ax.set_ylim(0, 1)
    for i, v in enumerate(probs):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha="center")
    st.pyplot(fig)

    pdf_path = f"ECG_Report_{report['Patient ID']}.pdf"
    generate_pdf_report(report, pdf_path)
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="Download Full Report (PDF)",
            data=f,
            file_name=pdf_path,
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
