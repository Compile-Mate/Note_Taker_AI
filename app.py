from flask import Flask, request, jsonify
import spacy
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load scispaCy medical model
try:
    nlp = spacy.load("en_core_sci_sm")  # Updated model
except Exception as e:
    print(f"Error loading scispaCy model: {e}")
    nlp = None  # Handle missing model case

# Load DistilBERT model for sentiment analysis
tokenizer = DistilBertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = DistilBertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

@app.route('/')
def index():
    return "Welcome to the Physician Notetaker API!"

# Extract medical details using scispaCy
def extract_medical_details(transcript):
    if not nlp:
        return {"error": "Medical NLP model not loaded"}
    
    doc = nlp(transcript)
    entities = {"Diseases": [], "Medications": []}

    for ent in doc.ents:
        if ent.label_.lower() in ["disease", "disorder"]:
            entities["Diseases"].append(ent.text)
        elif ent.label_.lower() in ["drug", "medication"]:
            entities["Medications"].append(ent.text)

    return {
        "Patient_Name": "Unknown",
        "Diseases": list(set(entities["Diseases"])),
        "Medications": list(set(entities["Medications"])),
        "Current_Status": "Stable",
        "Prognosis": "Follow-up recommended"
    }

# Sentiment & Intent Analysis
def analyze_sentiment_intent(patient_text):
    if not patient_text:
        return {"Sentiment": "Neutral", "Intent": "General Inquiry"}

    try:
        inputs = tokenizer(patient_text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        logits = outputs.logits
        sentiment = torch.argmax(logits, dim=1).item()
    except Exception as e:
        return {"Sentiment": "Error", "Intent": "Processing Failed", "Error": str(e)}

    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}

    intent = "Neutral"
    if "worry" in patient_text.lower() or "concerned" in patient_text.lower():
        intent = "Seeking reassurance"
    elif "pain" in patient_text.lower() or "discomfort" in patient_text.lower():
        intent = "Reporting symptoms"

    return {"Sentiment": sentiment_map.get(sentiment, "Unknown"), "Intent": intent}

# Generate SOAP Note
def generate_soap_note(transcript):
    medical_summary = extract_medical_details(transcript)
    return {
        "Subjective": {
            "Chief_Complaint": medical_summary["Diseases"],
            "History_of_Present_Illness": transcript
        },
        "Objective": {
            "Physical_Exam": "No significant abnormalities detected.",
            "Observations": "Patient stable."
        },
        "Assessment": {
            "Diagnosis": medical_summary["Diseases"],
            "Severity": "Mild"
        },
        "Plan": {
            "Medications": medical_summary["Medications"],
            "Follow-Up": medical_summary["Prognosis"]
        }
    }

# API Endpoint
@app.route('/process_transcript', methods=['POST'])
def process_transcript():
    data = request.get_json()
    if not data or 'transcript' not in data:
        return jsonify({"error": "No transcript provided"}), 400

    transcript = data['transcript']
    patient_text = data.get('patient_text', "")

    result = {
        "Medical_Summary": extract_medical_details(transcript),
        "Sentiment_Intent": analyze_sentiment_intent(patient_text),
        "SOAP_Note": generate_soap_note(transcript)
    }
    return jsonify(result)

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
