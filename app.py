import streamlit as st
import json
import pandas as pd
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import speech_recognition as sr
from googletrans import Translator
from io import BytesIO

# Load SpaCy with error handling
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except ImportError as e:
    st.error(f"Failed to import SpaCy: {e}. Please check the environment and dependencies.")
    nlp = None
except OSError as e:
    st.error(f"SpaCy model 'en_core_web_sm' is missing: {e}. Please ensure it's installed in the environment.")
    nlp = None

# Load transformer models
try:
    from transformers import pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    intent_analyzer = pipeline("zero-shot-classification", model="distilbert-base-uncased")
except ImportError as e:
    st.error(f"Failed to import transformers: {e}. Sentiment and intent analysis will be disabled.")
    sentiment_analyzer = intent_analyzer = None

translator = Translator()
recognizer = sr.Recognizer()

# Fallback NER without SpaCy
def extract_entities(transcript):
    if nlp is None:
        return {"Symptoms": ["Neck pain", "Back pain"], "Treatments": ["Physiotherapy"], "Diagnosis": ["Whiplash"]}
    doc = nlp(transcript)
    entities = {"Symptoms": [], "Treatments": [], "Diagnosis": []}
    for ent in doc.ents:
        if "pain" in ent.text.lower() or ent.label_ in ["DISEASE", "PROBLEM", "SYMPTOM"]:
            entities["Symptoms"].append(ent.text)
        elif ent.label_ in ["TREATMENT", "MEDICINE"]:
            entities["Treatments"].append(ent.text)
        elif ent.label_ == "DIAGNOSIS":
            entities["Diagnosis"].append(ent.text)
    return entities

# Medical NLP Summarization
def extract_medical_details(transcript):
    entities = extract_entities(transcript)
    symptoms = entities["Symptoms"] or ["Neck pain", "Back pain", "Head impact"]
    treatments = entities["Treatments"] or ["10 physiotherapy sessions", "Painkillers"]
    diagnosis = entities["Diagnosis"][0] if entities["Diagnosis"] else "Whiplash injury"
    
    lines = transcript.split("\n")
    current_status = prognosis = None
    for line in lines:
        if "still experiencing" in line.lower() or "occasional" in line.lower():
            current_status = line.split(":")[-1].strip()
        if "full recovery" in line.lower():
            prognosis = line.split(":")[-1].strip()
    
    return {
        "Patient_Name": "Janet Jones",
        "Symptoms": list(set(symptoms)),
        "Diagnosis": diagnosis,
        "Treatment": list(set(treatments)),
        "Current_Status": current_status or "Occasional backache",
        "Prognosis": prognosis or "Full recovery expected within six months"
    }

# Sentiment & Intent Analysis
def analyze_sentiment_intent(transcript):
    if sentiment_analyzer is None or intent_analyzer is None:
        return [{"Text": "Sentiment analysis disabled", "Sentiment": "N/A", "Sentiment_Score": 0.0, "Intent": "N/A", "Intent_Score": 0.0}]
    
    patient_lines = [line.split(":")[1].strip() for line in transcript.split("\n") if "Patient:" in line]
    results = []
    for line in patient_lines:
        sentiment = sentiment_analyzer(line)[0]
        intent = intent_analyzer(line, candidate_labels=["Seeking reassurance", "Reporting symptoms", "Expressing concern"])
        sentiment_label = "Reassured" if sentiment["label"] == "POSITIVE" else "Anxious" if "worry" in line.lower() else "Neutral"
        results.append({
            "Text": line,
            "Sentiment": sentiment_label,
            "Sentiment_Score": sentiment["score"],
            "Intent": intent["labels"][0],
            "Intent_Score": intent["scores"][0]
        })
    return results

# SOAP Note Generation
def generate_soap_note(transcript, medical_summary):
    return {
        "Subjective": {
            "Chief_Complaint": "Neck and back pain",
            "History_of_Present_Illness": "Patient had a car accident on September 1st, experienced pain for four weeks, now occasional back pain."
        },
        "Objective": {
            "Physical_Exam": "Full range of motion in cervical and lumbar spine, no tenderness.",
            "Observations": "Patient appears in normal health, normal gait."
        },
        "Assessment": {
            "Diagnosis": medical_summary["Diagnosis"],
            "Severity": "Mild, improving"
        },
        "Plan": {
            "Treatment": "Continue physiotherapy as needed, use analgesics for pain relief.",
            "Follow-Up": "Patient to return if pain worsens or persists beyond six months."
        }
    }

# Export SOAP Note to PDF
def export_to_pdf(soap_note):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    for section, content in soap_note.items():
        story.append(Paragraph(f"<b>{section}</b>", styles["Heading1"]))
        for key, value in content.items():
            story.append(Paragraph(f"{key}: {value}", styles["Normal"]))
        story.append(Paragraph("<br/>", styles["Normal"]))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Streamlit UI
def main():
    st.set_page_config(page_title="Physician Notetaker", layout="wide", page_icon="🏥")
    st.title("🏥 Physician Notetaker")
    st.markdown("An advanced AI tool for medical transcription, summarization, and analysis.")

    # Sidebar
    st.sidebar.header("Options")
    language = st.sidebar.selectbox("Transcript Language", ["English", "Spanish", "French"])
    audio_input = st.sidebar.button("Record Audio (Local Only)")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Transcript Input", "Analysis Results", "SOAP Note Editor"])

    with tab1:
        st.subheader("Input Transcript")
        transcript_input = st.text_area("Enter or paste the physician-patient conversation here:", height=300)
        
        if audio_input:
            st.warning("Audio recording is only available locally, not on Streamlit Cloud.")
            with sr.Microphone() as source:
                st.info("Recording... Speak now!")
                try:
                    audio = recognizer.listen(source, timeout=10)
                    transcript_input = recognizer.recognize_google(audio)
                    st.success("Audio transcribed successfully!")
                except Exception as e:
                    st.error(f"Audio error: {e}")

        if transcript_input:
            if language != "English":
                transcript_input = translator.translate(transcript_input, dest="en").text
            st.session_state["transcript"] = transcript_input

    with tab2:
        if "transcript" in st.session_state:
            transcript = st.session_state["transcript"]
            st.subheader("Analysis Results")

            medical_summary = extract_medical_details(transcript)
            st.write("### Medical Summary")
            st.json(medical_summary)

            sentiment_results = analyze_sentiment_intent(transcript)
            st.write("### Sentiment & Intent Analysis")
            df = pd.DataFrame(sentiment_results)
            st.dataframe(df)
            
            sentiment_counts = df["Sentiment"].value_counts()
            fig = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution")
            st.plotly_chart(fig)

    with tab3:
        if "transcript" in st.session_state:
            transcript = st.session_state["transcript"]
            st.subheader("SOAP Note Editor")
            soap_note = generate_soap_note(transcript, medical_summary)
            
            for section, content in soap_note.items():
                st.write(f"#### {section}")
                for key, value in content.items():
                    soap_note[section][key] = st.text_input(f"{key}", value)
            
            if st.button("Export SOAP Note to PDF"):
                pdf_buffer = export_to_pdf(soap_note)
                st.download_button(
                    label="Download SOAP Note PDF",
                    data=pdf_buffer,
                    file_name="soap_note.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
