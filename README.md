# Hybrid Paraphrase Detection System

A novel paraphrase detection system that combines **Siamese SBERT** with **Lexical Features** for detecting semantic similarity within PDF documents.

## 🎯 Overview

This system goes beyond simple plagiarism detection (exact match) to identify **semantic paraphrasing** where words differ but meaning remains the same. It uses a hybrid approach combining deep learning with traditional lexical analysis.

## 🔬 Novel Hybrid Architecture

The system implements a **three-pronged feature extraction approach**:

1. **SBERT (Sentence-BERT)** - Siamese Network Architecture
   - Uses pre-trained `paraphrase-MiniLM-L6-v2` model
   - Generates dense semantic embeddings for sentences
   - Computes cosine similarity between embeddings

2. **Jaccard Similarity** - Lexical Feature #1
   - Measures word-level overlap between sentences
   - Captures surface-level similarity

3. **Edit Distance** - Lexical Feature #2
   - Computes Levenshtein distance
   - Captures character-level similarity

These three features are combined using a **supervised Logistic Regression classifier** to make final paraphrase decisions.

## 🏗️ Technical Architecture

```
PDF Document
    ↓
[PDF Extractor - PyMuPDF]
    ↓
Raw Text
    ↓
[Text Preprocessor - spaCy]
    ↓
Sentence List
    ↓
[Hybrid Feature Extractor]
    ├── SBERT Embeddings → Cosine Similarity
    ├── Jaccard Similarity
    └── Edit Distance Similarity
    ↓
Feature Vector [cos_sim, jaccard_sim, edit_sim]
    ↓
[Supervised Classifier - Logistic Regression]
    ↓
Paraphrase Detection Results
    ↓
[Streamlit UI]
```

## 📦 Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

## 🚀 Usage

### Running the Application

```bash
streamlit run main.py
```

This will launch the web interface at `http://localhost:8501`

### Using the Interface

1. **Upload PDF**: Click "Browse files" and upload your PDF document
2. **View Sentences**: Expand "View Extracted Sentences" to see all extracted sentences
3. **Adjust Threshold**: Use the sidebar slider to control detection sensitivity
   - Higher threshold (0.8-1.0) = More conservative, fewer false positives
   - Lower threshold (0.5-0.7) = More lenient, may catch more paraphrases
4. **View Results**: 
   - **Summary Table**: Quick overview with all scores
   - **Detailed View**: Side-by-side comparison of paraphrase pairs

### Example Output

```
Paraphrase Pair #1
├── Sentence 15: "The company announced significant growth in revenue."
├── Sentence 42: "Revenue experienced substantial increases, the firm reported."
├── Overall Score: 0.872
├── SBERT Cosine: 0.891
├── Jaccard: 0.450
└── Edit Distance: 0.623
```

## 🔧 Component Details

### 1. PDF Extraction (`PDFExtractor`)
- Uses PyMuPDF (fitz) for reliable text extraction
- Handles multi-page documents
- Preserves text structure

### 2. Text Preprocessing (`TextPreprocessor`)
- Sentence segmentation using spaCy's NLP pipeline
- Text cleaning (whitespace normalization)
- Filters out very short sentences (< 5 words)

### 3. Hybrid Feature Extraction (`HybridFeatureExtractor`)
- **SBERT**: Generates 384-dimensional embeddings
- **Jaccard**: Set-based word overlap calculation
- **Edit Distance**: Normalized Levenshtein distance
- Returns combined feature vector

### 4. Classification (`ParaphraseClassifier`)
- Logistic Regression with StandardScaler
- Can be trained on labeled data (if available)
- Falls back to threshold-based approach using cosine similarity
- Supports model saving/loading

### 5. Main Engine (`ParaphraseDetector`)
- Orchestrates all components
- Generates all pairwise sentence comparisons
- Returns ranked paraphrase pairs

## 📊 Feature Interpretation

- **Overall Score** (0-1): Classifier's confidence that sentences are paraphrases
- **SBERT Cosine** (0-1): Semantic similarity from deep learning
- **Jaccard** (0-1): Lexical word overlap
- **Edit Distance** (0-1): Character-level similarity

High scores across all metrics indicate strong paraphrase relationship.

## 🎓 Academic Requirements Met

✅ **PDF Integration**: PyMuPDF for text extraction  
✅ **Sentence-Level Analysis**: spaCy segmentation  
✅ **Semantic Matching**: SBERT embeddings  
✅ **Frontend Interface**: Streamlit application  
✅ **NLP Framework**: spaCy for preprocessing  
✅ **BERT/SBERT**: Sentence-BERT model  
✅ **Siamese Network**: Shared-weight twin encoders in SBERT  
✅ **Supervised Learning**: Logistic Regression classifier  
✅ **Novel Hybrid**: SBERT + Jaccard + Edit Distance combination  

## 🔬 Model Training (Optional)

The system works out-of-the-box with a threshold-based approach. To train the classifier on labeled data:

```python
# Prepare training data
X_train = np.array([...])  # Feature vectors
y_train = np.array([...])  # Labels (1=paraphrase, 0=not)

# Train classifier
detector = ParaphraseDetector()
detector.classifier.train(X_train, y_train)

# Save model
detector.classifier.save_model('paraphrase_model.pkl')
```

## 📝 Dependencies

- **streamlit**: Web interface
- **PyMuPDF**: PDF text extraction
- **spacy**: NLP preprocessing
- **sentence-transformers**: SBERT embeddings
- **scikit-learn**: Machine learning
- **python-Levenshtein**: Edit distance computation
- **numpy, pandas**: Data handling

## 🔍 Performance Considerations

- **Computation**: O(n²) comparisons for n sentences
- **Memory**: Scales with document size
- **Optimization**: Batch processing for embeddings
- For large documents (>100 sentences), consider filtering or sampling strategies

## 🛠️ Customization

### Change SBERT Model
Edit `HybridFeatureExtractor.__init__()`:
```python
self.sbert_model = SentenceTransformer('paraphrase-mpnet-base-v2')  # More accurate
```

### Adjust Feature Weights
Modify `extract_hybrid_features()` to apply custom weights to features.

### Add More Features
Extend with additional features:
- WordNet synset overlap
- POS tag similarity
- Dependency parse overlap
- TF-IDF scores

## 📄 License

This is an academic project demonstrating hybrid paraphrase detection techniques.

## 🤝 Contributing

Feel free to extend with:
- Additional lexical features
- Knowledge graph integration (WordNet)
- Advanced classifiers (Neural Networks)
- Batch processing for large documents

## 📧 Support

For issues or questions, please check the code comments or raise an issue.

---

**Built with ❤️ for advanced NLP research**
