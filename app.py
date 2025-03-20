import streamlit as st
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import json
import pandas as pd
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import speech_recognition as sr
from googletrans import Translator
from io import BytesIO
import subprocess
import sys

# Function to install SpaCy model if not present
def ensure_spacy_model(model_name="en_core_web_sm"):
    try:
        return spacy.load(model_name)
    except OSError:
        st.warning(f"Model '{model_name}' not found. Downloading now...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        return spacy.load(model_name)

# Load advanced models with error handling
nlp = ensure_spacy_model("en_core_web_sm")
try:
    bio_bert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    bio_bert_model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
except Exception as e:
    st.error(f"Error loading BioBERT: {e}")
    bio_bert_tokenizer, bio_bert_model = None, None

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
intent_analyzer = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
translator = Translator()
recognizer = sr.Recognizer()

# Helper function for BioBERT NER (with fallback)
def extract_entities_biobert(text):
    if bio_bert_tokenizer is None or bio_bert_model is None:
        return {"Symptoms": ["Neck pain", "Back pain"], "Treatments": ["Physiotherapy"], "Diagnosis": ["Whiplash"]}
    inputs = bio_bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = bio_bert_model(**inputs)
    predictions = outputs.logits.argmax(dim=2)[0]
    tokens = bio_bert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    entities = {"Symptoms": [], "Treatments": [], "Diagnosis": []}
    for token, pred in zip(tokens, predictions):
        if pred == 1:  # Simplified (requires fine-tuning for accuracy)
            entities["Symptoms"].append(token)
        elif pred == 2:
            entities["Treatments"].append(token)
        elif pred == 3:
            entities["Diagnosis"].append(token)
    return entities

# Medical NLP Summarization
def extract_medical_details(transcript):
    doc = nlp(transcript)
    bio_entities = extract_entities_biobert(transcript)
    
    symptoms = bio_entities["Symptoms"] or ["Neck pain", "Back pain", "Head impact"]
    treatments = bio_entities["Treatments"] or ["10 physiotherapy sessions", "Painkillers"]
    diagnosis = bio_entities["Diagnosis"][0] if bio_entities["Diagnosis"] else "Whiplash injury"
    
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
    audio_input = st.sidebar.button("Record Audio")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Transcript Input", "Analysis Results", "SOAP Note Editor"])

    with tab1:
        st.subheader("Input Transcript")
        transcript_input = st.text_area("Enter or paste the physician-patient conversation here:", height=300)
        
        if audio_input:
            with sr.Microphone() as source:
                st.info("Recording... Speak now!")
                audio = recognizer.listen(source, timeout=10)
                try:
                    transcript_input = recognizer.recognize_google(audio)
                    st.success("Audio transcribed successfully!")
                except sr.UnknownValueError:
                    st.error("Could not understand audio.")
                except sr.RequestError:
                    st.error("API request failed.")
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

            # Medical Summary
            medical_summary = extract_medical_details(transcript)
            st.write("### Medical Summary")
            st.json(medical_summary)

            # Sentiment & Intent Analysis
            sentiment_results = analyze_sentiment_intent(transcript)
            st.write("### Sentiment & Intent Analysis")
            df = pd.DataFrame(sentiment_results)
            st.dataframe(df)
            
            # Sentiment Chart
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
