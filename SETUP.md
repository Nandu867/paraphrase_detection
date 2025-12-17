# Setup Instructions for Hybrid Paraphrase Detection System

## Quick Start Guide

Follow these steps to set up and run the application:

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- streamlit (Web UI)
- PyMuPDF (PDF extraction)
- spacy (NLP preprocessing)
- sentence-transformers (SBERT embeddings)
- scikit-learn (ML classifier)
- python-Levenshtein (Edit distance)
- numpy, pandas, torch, transformers

### 2. Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

This downloads the English language model needed for sentence segmentation.

### 3. Run the Application

```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

### 4. Test the System

1. Prepare a PDF file with some text (ideally with some paraphrased content)
2. Upload the PDF through the web interface
3. Adjust the similarity threshold (default: 0.7)
4. View detected paraphrases in the results

## Troubleshooting

### Issue: spaCy model not found
**Solution**: Run `python -m spacy download en_core_web_sm`

### Issue: PyTorch installation fails
**Solution**: Install PyTorch separately first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Memory error with large PDFs
**Solution**: Process smaller documents or increase system memory

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Internet connection for first-time model downloads

## Next Steps

- See README.md for detailed documentation
- Customize the model by editing main.py
- Add more features or change thresholds as needed
