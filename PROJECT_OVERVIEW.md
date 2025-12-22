# Hybrid Paraphrase Detection System - Project Overview

## 🎯 Project Summary

This project implements a **Hybrid Paraphrase Detection System** that combines deep semantic embeddings (Siamese SBERT) with lexical features to detect paraphrased sentences in PDF documents.

---

## 🏗️ Project Architecture

```
Paraphrase Detection System
├── Data Input (PDF)
│   ├── PDF Text Extraction
│   └── Sentence Splitting (spaCy)
├── Feature Extraction (Hybrid)
│   ├── SBERT Embeddings (Siamese Network)
│   ├── Jaccard Similarity (Lexical)
│   └── Edit Distance (Character-level)
├── Classification
│   └── Logistic Regression (trained on Quora data)
└── Output (Paraphrase Pairs + Scores)
```

---

## 📋 What We Did

### 1. **Fixed Dependency Issues**
- Installed all required Python packages with Python 3.11.9
- Resolved import errors (sentence-transformers, sklearn, spacy, etc.)

### 2. **Created Model Training Pipeline**
- Implemented `train.py` to train Logistic Regression on 50,000 Quora question pairs
- Split data: 40,000 training samples + 10,000 test samples
- Extracted hybrid features from question pairs

### 3. **Built Web Application**
- Created `main.py` - Streamlit web interface
- Allows users to upload PDF files
- Displays paraphrase detection results in real-time

### 4. **Fixed Model Loading Issues**
- Updated save/load logic to include `is_trained` flag
- Handles corrupt model files gracefully
- Auto-retrains if model is invalid

### 5. **Optimized Results Display**
- Added scrollable DataFrame for results
- Implemented tabs (Summary Table + Detailed View)
- Added CSV download functionality

---

## 📊 Training Results

```
Dataset: Quora Question Pairs (50,000 pairs)
Train/Test Split: 80/20 (40,000 / 10,000)

Performance Metrics:
├── Test Accuracy:  79.03%
├── Test Precision: 70.11%
├── Test Recall:    76.33%
└── Test F1-Score:  73.08%

Features Used:
├── SBERT Cosine Similarity (Semantic)
├── Jaccard Similarity (Lexical)
└── Edit Distance Similarity (Character-level)
```

---

## 🔧 Technology Stack

| Component | Technology |
|-----------|-----------|
| **Semantic Embeddings** | Sentence-BERT (paraphrase-MiniLM-L6-v2) |
| **Architecture** | Siamese Network |
| **Classifier** | Logistic Regression (scikit-learn) |
| **Text Processing** | spaCy (sentence splitting) |
| **PDF Extraction** | PyMuPDF (fitz) |
| **Web Framework** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Evaluation** | scikit-learn metrics |

---

## 📁 File Structure

```
Paraphrase Detection/
├── main.py                          # Streamlit web application
├── train.py                         # Training pipeline script
├── requirements.txt                 # Python dependencies
├── paraphrase_classifier.pkl        # Trained model (binary)
├── quora_duplicate_questions.csv    # Training dataset
├── README.md                        # Basic documentation
├── .venv/                          # Virtual environment
└── .git/                           # Git version control
```

---

## 🎯 How It Works - Step by Step

### **Step 1: PDF Upload & Text Extraction**
```
User uploads PDF
    ↓
PDFExtractor.extract_text_from_pdf()
    ↓
Raw text extracted
```

### **Step 2: Sentence Splitting**
```
Raw text
    ↓
TextPreprocessor.split_into_sentences()
    ↓
List of sentences (using spaCy)
```

### **Step 3: Feature Extraction**
```
Sentence Pairs
    ↓
HybridFeatureExtractor.extract_features()
    ├── SBERT embeddings → Cosine Similarity
    ├── Word tokens → Jaccard Similarity
    └── Character distance → Edit Distance Similarity
    ↓
Feature vectors (3 dimensions)
```

### **Step 4: Classification**
```
Feature vectors
    ↓
Trained Logistic Regression classifier
    ↓
Probability score (0.0 - 1.0)
```

### **Step 5: Results Output**
```
Paraphrase pairs detected
    ↓
Display in Streamlit UI with:
├── Sentence indices
├── Overall score
├── Individual feature scores
└── CSV download option
```

---

## 📤 Output Format

### **Summary Table Display**
```
| Sentence 1 (Index) | Sentence 2 (Index) | Overall Score | SBERT Cosine | Jaccard | Edit Distance |
|-------------------|-------------------|---------------|--------------|---------|---------------|
| 0                 | 34                | 0.711         | 0.866        | 0.333   | 0.443        |
| 5                 | 12                | 0.745         | 0.878        | 0.421   | 0.512        |
```

### **Detailed View Display**
Shows pair-by-pair analysis with:
- Full sentence text
- Feature breakdown
- Classification score
- Visual highlighting

### **CSV Export**
All results saved with columns:
- Sentence 1 Index
- Sentence 2 Index
- Overall Score
- SBERT Cosine
- Jaccard Similarity
- Edit Distance

---

## 🚀 How to Use

### **1. Train the Model** (One-time setup)
```bash
python train.py
```
- Loads Quora dataset
- Generates features for 50,000 pairs
- Trains Logistic Regression classifier
- Saves model to `paraphrase_classifier.pkl`

### **2. Run Web Application**
```bash
streamlit run main.py
```
- Opens at `http://localhost:8501`
- Loads trained model automatically
- Ready for PDF uploads

### **3. Upload PDF**
- Click "Browse files" in web interface
- Select PDF document
- System extracts text and detects paraphrases

### **4. View Results**
- See detected paraphrase pairs
- Check individual feature scores
- Download results as CSV

---

## 🔍 Feature Explanation

### **SBERT Cosine Similarity** (Semantic)
- Uses Siamese network architecture
- Generates 384-dimensional embeddings
- Measures semantic meaning similarity
- Range: 0 (completely different) to 1 (identical meaning)

### **Jaccard Similarity** (Lexical)
- Counts shared words between sentences
- Formula: `intersection / union`
- Measures lexical word overlap
- Range: 0 (no shared words) to 1 (identical words)

### **Edit Distance Similarity** (Character-level)
- Levenshtein distance normalized
- Measures character-level changes needed
- Captures typos and variations
- Range: 0 to 1 (normalized)

### **Hybrid Approach**
- Combines all 3 features
- Logistic Regression learns optimal weights
- Better than single feature alone

---

## 📈 Performance Characteristics

**Strengths:**
- High semantic understanding (SBERT)
- Captures lexical differences (Jaccard)
- Detects minor variations (Edit Distance)
- Fast inference (~100ms per pair)

**Limitations:**
- Requires training data (50,000 pairs)
- Language-specific (English model)
- CPU usage for embeddings
- Memory intensive with large documents

---

## 🔧 Configuration

### **Similarity Threshold**
- Default: 0.70
- Adjustable via slider in UI
- Higher = stricter detection
- Lower = more lenient detection

### **Model Parameters**
- SBERT Model: `paraphrase-MiniLM-L6-v2`
- Classifier: Logistic Regression
- Scaler: StandardScaler
- Random State: 42 (reproducible)

---

## 📦 Dependencies

```
streamlit>=1.28.0           # Web interface
PyMuPDF>=1.23.0            # PDF extraction
spacy>=3.7.0               # NLP & sentence splitting
sentence-transformers>=2.2.0 # SBERT embeddings
numpy>=1.24.0              # Numerical operations
pandas>=2.0.0              # Data manipulation
scikit-learn>=1.3.0        # ML algorithms & metrics
python-Levenshtein>=0.21.0 # Edit distance
torch>=2.0.0               # Deep learning
transformers>=4.30.0       # Hugging Face models
datasets>=2.14.0           # Dataset loading
```

---

## 🎓 Learning Outcomes

This project demonstrates:
1. **Siamese Neural Networks** - Shared weight architecture
2. **Hybrid Feature Engineering** - Combining multiple approaches
3. **Model Training & Evaluation** - scikit-learn workflow
4. **Web Application Development** - Streamlit framework
5. **NLP Best Practices** - Text preprocessing, embeddings
6. **Model Deployment** - Pickle serialization, loading

---

## 📝 Example Output

**Input:** PDF with sentences:
- "What is machine learning?"
- "Define machine learning"
- "How to cook pasta?"

**Processing:**
1. Extract 3 sentences
2. Compute 3 pairs: (0,1), (0,2), (1,2)
3. Extract features for each pair

**Output:**
```
✅ Found 1 potential paraphrase pair

Sentence 0 & 1: Score 0.82 (PARAPHRASE ✓)
- SBERT: 0.89 (highly semantic match)
- Jaccard: 0.75 (good word overlap)
- Edit Distance: 0.70 (similar length)

Sentence 0 & 2: Score 0.15 (NOT paraphrase)
- SBERT: 0.18 (different topics)
- Jaccard: 0.10 (minimal overlap)
- Edit Distance: 0.08 (very different)
```

---

## ✅ What's Complete

- ✅ Hybrid feature extraction (SBERT + Jaccard + Edit Distance)
- ✅ Model training on Quora dataset (79% accuracy)
- ✅ Web interface for PDF processing
- ✅ Real-time paraphrase detection
- ✅ Results visualization with scores
- ✅ CSV export functionality
- ✅ Threshold adjustment
- ✅ Model persistence and loading

---

## 🚀 Ready to Use!

The system is fully functional and ready for detecting paraphrases in PDF documents. Simply:
1. Start the app: `streamlit run main.py`
2. Upload a PDF
3. Review paraphrase detection results
4. Download results as CSV

Enjoy! 🎉
