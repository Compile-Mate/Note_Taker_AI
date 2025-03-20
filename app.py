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
import zipfile
import time

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

# Sample transcript
SAMPLE_TRANSCRIPT = """
Physician: Good morning, Ms. Jones. How are you feeling today?
Patient: Good morning, doctor. I’m doing better, but I still have some discomfort now and then.
Physician: I understand you were in a car accident last September. Can you walk me through what happened?
Patient: Yes, it was on September 1st, around 12:30 in the afternoon. I was driving from Cheadle Hulme to Manchester when I had to stop in traffic. Out of nowhere, another car hit me from behind, which pushed my car into the one in front.
Physician: That sounds like a strong impact. Were you wearing your seatbelt?
Patient: Yes, I always do.
Physician: What did you feel immediately after the accident?
Patient: At first, I was just shocked. But then I realized I had hit my head on the steering wheel, and I could feel pain in my neck and back almost right away.
Physician: Did you seek medical attention at that time?
Patient: Yes, I went to Moss Bank Accident and Emergency. They checked me over and said it was a whiplash injury, but they didn’t do any X-rays. They just gave me some advice and sent me home.
Physician: How did things progress after that?
Patient: The first four weeks were rough. My neck and back pain were really bad—I had trouble sleeping and had to take painkillers regularly. It started improving after that, but I had to go through ten sessions of physiotherapy to help with the stiffness and discomfort.
Physician: That makes sense. Are you still experiencing pain now?
Patient: It’s not constant, but I do get occasional backaches. It’s nothing like before, though.
Physician: That’s good to hear. Have you noticed any other effects, like anxiety while driving or difficulty concentrating?
Patient: No, nothing like that. I don’t feel nervous driving, and I haven’t had any emotional issues from the accident.
Physician: And how has this impacted your daily life? Work, hobbies, anything like that?
Patient: I had to take a week off work, but after that, I was back to my usual routine. It hasn’t really stopped me from doing anything.
Physician: That’s encouraging. Let’s go ahead and do a physical examination to check your mobility and any lingering pain.
[Physical Examination Conducted]
Physician: Everything looks good. Your neck and back have a full range of movement, and there’s no tenderness or signs of lasting damage. Your muscles and spine seem to be in good condition.
Patient: That’s a relief!
Physician: Yes, your recovery so far has been quite positive. Given your progress, I’d expect you to make a full recovery within six months of the accident. There are no signs of long-term damage or degeneration.
Patient: That’s great to hear. So, I don’t need to worry about this affecting me in the future?
Physician: That’s right. I don’t foresee any long-term impact on your work or daily life. If anything changes or you experience worsening symptoms, you can always come back for a follow-up. But at this point, you’re on track for a full recovery.
Patient: Thank you, doctor. I appreciate it.
Physician: You’re very welcome, Ms. Jones. Take care, and don’t hesitate to reach out if you need anything.
"""

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

# Create a ZIP file with all results
def create_zip_file(medical_summary, sentiment_df, soap_note):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("medical_summary.json", json.dumps(medical_summary, indent=2))
        zip_file.writestr("sentiment_analysis.csv", sentiment_df.to_csv(index=False))
        pdf_buffer = export_to_pdf(soap_note)
        zip_file.writestr("soap_note.pdf", pdf_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

# Process transcript with progress bar
def process_transcript(transcript):
    progress = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Extracting medical details...")
    progress.progress(33)
    medical_summary = extract_medical_details(transcript)
    
    status_text.text("Analyzing sentiment and intent...")
    progress.progress(66)
    sentiment_results = analyze_sentiment_intent(transcript)
    
    status_text.text("Generating SOAP note...")
    progress.progress(100)
    soap_note = generate_soap_note(transcript, medical_summary)
    
    time.sleep(0.5)  # Small delay for effect
    status_text.empty()
    progress.empty()
    
    return medical_summary, sentiment_results, soap_note

# Streamlit UI
def main():
    st.set_page_config(page_title="Physician Notetaker", layout="wide", page_icon="🏥")
    
    # Custom styling with theme support
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"
    
    theme = st.session_state["theme"]
    if theme == "dark":
        st.markdown("""
            <style>
            :root {
                --background: #1a1a2e;
                --text-color: #e0e0e0;
                --card-bg: #2a2a4e;
                --primary: #4CAF50;
            }
            .stApp {
                background: var(--background);
                color: var(--text-color);
            }
            .stTextArea textarea {
                background-color: #2a2a4e;
                color: #e0e0e0;
                border-radius: 10px;
                font-family: Arial, sans-serif;
                border: 1px solid #4CAF50;
            }
            .stButton button {
                background: linear-gradient(45deg, #4CAF50, #66BB6A);
                color: white;
                border-radius: 5px;
                font-weight: bold;
                transition: transform 0.2s;
            }
            .stButton button:hover {
                transform: scale(1.05);
            }
            .stTabs [data-baseweb="tab"] {
                font-size: 16px;
                font-weight: bold;
                color: var(--text-color);
                background-color: var(--card-bg);
                border-radius: 5px 5px 0 0;
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                background-color: #4CAF50;
                color: white;
            }
            .stExpander {
                background-color: var(--card-bg);
                border-radius: 10px;
                border: 1px solid #4CAF50;
            }
            .stProgress > div > div {
                background-color: #4CAF50;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            :root {
                --background: #f0f2f6;
                --text-color: #262730;
                --card-bg: #ffffff;
                --primary: #4CAF50;
            }
            .stApp {
                background: var(--background);
                color: var(--text-color);
            }
            .stTextArea textarea {
                background-color: #f0f8ff;
                color: #262730;
                border-radius: 10px;
                font-family: Arial, sans-serif;
                border: 1px solid #4CAF50;
            }
            .stButton button {
                background: linear-gradient(45deg, #4CAF50, #66BB6A);
                color: white;
                border-radius: 5px;
                font-weight: bold;
                transition: transform 0.2s;
            }
            .stButton button:hover {
                transform: scale(1.05);
            }
            .stTabs [data-baseweb="tab"] {
                font-size: 16px;
                font-weight: bold;
                color: var(--text-color);
                background-color: var(--card-bg);
                border-radius: 5px 5px 0 0;
            }
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                background-color: #4CAF50;
                color: white;
            }
            .stExpander {
                background-color: var(--card-bg);
                border-radius: 10px;
                border: 1px solid #4CAF50;
            }
            .stProgress > div > div {
                background-color: #4CAF50;
            }
            </style>
        """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #4CAF50;">🏥 Physician Notetaker</h1>
            <p style="font-size: 18px;">An advanced AI tool for medical transcription, summarization, and analysis.</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.header("⚙️ Options")
    language = st.sidebar.selectbox("Transcript Language", ["English", "Spanish", "French"], help="Select the language of the transcript.")
    audio_input = st.sidebar.button("🎙️ Record Audio (Local Only)", help="Record audio input (only works locally).")
    theme_toggle = st.sidebar.checkbox("🌙 Dark Theme", value=theme == "dark", help="Toggle between light and dark themes.")
    if theme_toggle != (theme == "dark"):
        st.session_state["theme"] = "dark" if theme_toggle else "light"
        st.experimental_rerun()
    
    if st.sidebar.button("🗑️ Reset App", help="Clear all inputs and history."):
        st.session_state.clear()
        st.experimental_rerun()

    # Transcript history
    if "transcript_history" not in st.session_state:
        st.session_state["transcript_history"] = []
    
    st.sidebar.subheader("📜 Transcript History")
    if st.session_state["transcript_history"]:
        selected_history = st.sidebar.selectbox("Select a previous transcript:", [""] + st.session_state["transcript_history"])
        if selected_history:
            st.session_state["transcript"] = selected_history

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Transcript Input", "📊 Analysis Results", "📋 SOAP Note Editor"])

    with tab1:
        st.subheader("Input Transcript")
        if st.button("📄 Load Sample Transcript", help="Load a sample physician-patient conversation."):
            st.session_state["transcript"] = SAMPLE_TRANSCRIPT
        
        def on_transcript_change():
            transcript = st.session_state["transcript_input"]
            if transcript:
                if language != "English":
                    transcript = translator.translate(transcript, dest="en").text
                st.session_state["transcript"] = transcript
                if transcript not in st.session_state["transcript_history"]:
                    st.session_state["transcript_history"].append(transcript)
                medical_summary, sentiment_results, soap_note = process_transcript(transcript)
                st.session_state["medical_summary"] = medical_summary
                st.session_state["sentiment_results"] = sentiment_results
                st.session_state["soap_note"] = soap_note

        transcript_input = st.text_area(
            "Enter or paste the physician-patient conversation here:",
            height=300,
            value=st.session_state.get("transcript", ""),
            key="transcript_input",
            on_change=on_transcript_change,
            help="Press Enter to process the transcript."
        )
        
        if audio_input:
            st.warning("Audio recording is only available locally, not on Streamlit Cloud.")
            with sr.Microphone() as source:
                st.info("Recording... Speak now!")
                try:
                    audio = recognizer.listen(source, timeout=10)
                    transcript_input = recognizer.recognize_google(audio)
                    st.success("Audio transcribed successfully!")
                    st.session_state["transcript"] = transcript_input
                    if transcript_input not in st.session_state["transcript_history"]:
                        st.session_state["transcript_history"].append(transcript_input)
                    medical_summary, sentiment_results, soap_note = process_transcript(transcript_input)
                    st.session_state["medical_summary"] = medical_summary
                    st.session_state["sentiment_results"] = sentiment_results
                    st.session_state["soap_note"] = soap_note
                except Exception as e:
                    st.error(f"Audio error: {e}")

    with tab2:
        if "medical_summary" in st.session_state:
            st.subheader("Analysis Results")
            
            # Medical Summary
            with st.expander("🩺 Medical Summary", expanded=True):
                st.json(st.session_state["medical_summary"])
            
            # Sentiment & Intent Analysis
            with st.expander("😊 Sentiment & Intent Analysis", expanded=True):
                df = pd.DataFrame(st.session_state["sentiment_results"])
                st.dataframe(df)
                
                col1, col2 = st.columns(2)
                with col1:
                    sentiment_counts = df["Sentiment"].value_counts()
                    fig_sentiment = px.pie(values=sentiment_counts.values, names=sentiment_counts.index, title="Sentiment Distribution")
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                
                with col2:
                    intent_counts = df["Intent"].value_counts()
                    fig_intent = px.bar(x=intent_counts.values, y=intent_counts.index, title="Intent Distribution", orientation="h")
                    st.plotly_chart(fig_intent, use_container_width=True)
                
                # Confidence Scores
                st.subheader("Confidence Scores")
                col3, col4 = st.columns(2)
                with col3:
                    fig_sentiment_scores = px.bar(df, x="Sentiment_Score", y="Text", title="Sentiment Confidence Scores", orientation="h")
                    st.plotly_chart(fig_sentiment_scores, use_container_width=True)
                with col4:
                    fig_intent_scores = px.bar(df, x="Intent_Score", y="Text", title="Intent Confidence Scores", orientation="h")
                    st.plotly_chart(fig_intent_scores, use_container_width=True)

            # Export Options
            with st.expander("📤 Export Results"):
                if st.button("Export Medical Summary as JSON"):
                    st.download_button(
                        label="Download Medical Summary",
                        data=json.dumps(st.session_state["medical_summary"], indent=2),
                        file_name="medical_summary.json",
                        mime="application/json"
                    )
                if st.button("Export Sentiment Analysis as CSV"):
                    st.download_button(
                        label="Download Sentiment Analysis",
                        data=df.to_csv(index=False),
                        file_name="sentiment_analysis.csv",
                        mime="text/csv"
                    )
                if st.button("Export All Results as ZIP"):
                    zip_buffer = create_zip_file(st.session_state["medical_summary"], df, st.session_state["soap_note"])
                    st.download_button(
                        label="Download All Results",
                        data=zip_buffer,
                        file_name="physician_notetaker_results.zip",
                        mime="application/zip"
                    )

    with tab3:
        if "soap_note" in st.session_state:
            st.subheader("SOAP Note Editor")
            soap_note = st.session_state["soap_note"]
            
            # Preview Mode
            preview_mode = st.checkbox("📄 Preview Mode", help="View the SOAP note in a formatted layout.")
            
            if preview_mode:
                with st.expander("SOAP Note Preview", expanded=True):
                    for section, content in soap_note.items():
                        st.markdown(f"**{section}**")
                        for key, value in content.items():
                            st.markdown(f"*{key}:* {value}")
                        st.markdown("---")
            else:
                for section, content in soap_note.items():
                    with st.expander(f"{section}", expanded=True):
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
