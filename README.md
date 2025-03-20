# Physician Notetaker

An advanced AI-powered tool for medical transcription, summarization, and analysis, built with Streamlit. This app processes physician-patient conversations, extracts medical details, performs sentiment and intent analysis, and generates SOAP notes.

## 🚀 Live Demo
The app is live and functional on Streamlit Cloud:  
🔗 [Physician Notetaker Live URL](https://notetakerai.streamlit.app/)

---

## ✨ Features
- **Medical NLP Summarization:** Extracts symptoms, diagnosis, treatments, and prognosis using SpaCy.
- **Sentiment & Intent Analysis:** Analyzes patient emotions and intent using transformer models (DistilBERT).
- **SOAP Note Generation:** Creates structured SOAP notes with an editable interface and PDF export.
- **Multi-Language Support:** Supports English, Spanish, and French transcripts using `googletrans`.
- **Audio Transcription:** Supports local audio input using `speechrecognition` (not available on Streamlit Cloud).
- **Advanced UI:** Modern design with light/dark themes, progress bars, collapsible sections, and interactive charts.
- **Transcript History:** Stores and allows reprocessing of previous transcripts.
- **Export Options:** Export medical summary, sentiment analysis, and SOAP note as JSON, CSV, PDF, or a ZIP file.

---

## 🛠️ Setup and Running Instructions

### Prerequisites
- Python 3.11
- Git
- A Streamlit Cloud account (for deployment)

### 📌 Local Setup
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Compile-Mate/Note_Taker_AI.git
   cd Note_Taker_AI
   ```
2. **Create a Virtual Environment:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the App Locally:**
   ```bash
   streamlit run app.py
   ```
5. Open your browser to `http://localhost:8501` to view the app.

### 🚀 Deployment on Streamlit Cloud
1. **Fork or Push to GitHub:** Ensure the repository is public and contains `app.py`, `requirements.txt`, and this `README.md`.
2. **Deploy on Streamlit Cloud:**
   - Go to [Streamlit Cloud](https://share.streamlit.io/) and sign in.
   - Click "New App" > "From existing repo" and select your repository.
   - Set the main file to `app.py`.
   - Choose Python 3.11 as the Python version.
   - Click "Deploy" and wait for the app to build.
3. **Access the Live App:**
   - The app will be available at: [https://notetakerai.streamlit.app/](https://notetakerai.streamlit.app/)

---

## 📸 Sample Output Screenshots
### 📌 Transcript Input
The "Transcript Input" tab allows users to enter or load a sample physician-patient conversation. Pressing Enter processes the transcript immediately.

### 📊 Analysis Results
The "Analysis Results" tab displays the extracted medical summary, sentiment and intent analysis, distribution charts, and confidence scores for each analysis.

### 📝 SOAP Note Editor
The "SOAP Note Editor" tab provides an editable interface for the generated SOAP note, with a preview mode and PDF export option.

---

## 🔬 Methodologies Used

### 🏥 1. Medical NLP Summarization
- **Algorithm:** Used SpaCy (`en_core_web_sm`) for Named Entity Recognition (NER) to extract medical entities like symptoms, treatments, and diagnosis.
- **Reasoning:** SpaCy is lightweight and effective for general-purpose NER, making it suitable for Streamlit Cloud's memory constraints.

### 🤖 2. Sentiment & Intent Analysis
- **Algorithm:** Used `distilbert-base-uncased-finetuned-sst-2-english` for sentiment analysis and `distilbert-base-uncased` for zero-shot intent classification.
- **Reasoning:** DistilBERT balances accuracy and memory usage, which is critical for deployment.

### 📄 3. SOAP Note Generation
- **Algorithm:** Rule-based mapping of transcript sections to Subjective, Objective, Assessment, and Plan (SOAP) components.
- **Reasoning:** Ensures deterministic and structured output, essential for medical applications.

### 🎨 4. UI and Interactivity
- **Tools:** Streamlit for UI, Plotly for interactive charts, and custom CSS for styling.
- **Reasoning:** Streamlit allows rapid development with interactive features.

### 🏆 5. Additional Features
- **Multi-Language Support:** Uses `googletrans` to process English, Spanish, and French transcripts.
- **Audio Input:** Integrated `speechrecognition` for local audio transcription (not available on Streamlit Cloud).
- **Export Options:** Allows ZIP export bundling all results.
- **Transcript History:** Stores previous transcripts for easy access.
- **Real-Time Processing:** Uses progress bars and Enter-key processing for efficiency.

---

## ⚠️ Challenges and Solutions
- **Dependency Issues:** Fixed by pinning versions (`spacy==3.7.2`, `thinc==8.2.2`, `torch==2.3.1`) and using Python 3.11.
- **Memory Constraints:** Used DistilBERT instead of heavier models like BioBERT.
- **Audio Input Limitation:** Audio recording isn’t available on Streamlit Cloud due to microphone restrictions.
- **Enter Key Processing:** Modified the app to process transcripts on pressing Enter.
- **Model Download Issues:** Pre-specified model URLs in `requirements.txt` to avoid runtime download issues.

---

## 🔮 Future Improvements
✅ Fine-tune transformer models on medical datasets (e.g., MIMIC-III) for better accuracy.
✅ Deploy on a custom server (e.g., AWS) to support audio input, authentication, and higher memory limits.
✅ Add real-time collaboration features using a backend database (e.g., Firebase).
✅ Integrate with FHIR for interoperability with Electronic Health Records (EHR) systems.
✅ Enhance UI with accessibility features (keyboard navigation, screen reader support).

---

## ⭐ Contribute & Support
If you find this project helpful, please consider giving it a ⭐ on [GitHub](https://github.com/Compile-Mate/Note_Taker_AI)! 🎉
