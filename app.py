import streamlit as st
import tempfile
from ecg_utils import process_ecg_image, generate_pdf_report
from PIL import Image
import datetime

st.set_page_config(page_title="ECG MI & Type Detector", layout="centered")

st.title("🩺 ECG Myocardial Infarction & Type Detection")
st.write("Upload an ECG graph image to detect MI and classify its type (e.g., Anterior, Inferior).")

uploaded_file = st.file_uploader("Upload ECG Image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded ECG Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image.save(tmp_file.name)
        report, confidence = process_ecg_image(tmp_file.name)

    st.subheader("📋 Diagnosis")
    st.markdown(f"**Prediction:** {report['Diagnosis']} ({confidence*100:.2f}%)")
    st.markdown(f"**ST Segment Status:** {report['ST Segment']}")
    if report["Diagnosis"] == "Myocardial Infarction":
        st.markdown(f"**MI Type:** {report['MI Type']}")

    st.markdown("---")
    st.json(report)

    if st.button("📄 Generate PDF Report"):
        report["Test Date"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        generate_pdf_report(report)
        with open("ECG_Report.pdf", "rb") as f:
            st.download_button("📥 Download ECG Report", f, file_name="ECG_Report.pdf")
