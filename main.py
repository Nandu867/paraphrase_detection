"""
Hybrid Paraphrase Detection System
Combines Siamese SBERT with Lexical Features (Jaccard Similarity & Edit Distance)
"""

import streamlit as st
import fitz  # PyMuPDF
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import List, Tuple, Dict
import re
from itertools import combinations
import Levenshtein  # for edit distance
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. PDF TEXT EXTRACTION MODULE
# =============================================================================


class PDFExtractor:
    """Extract text from PDF files using PyMuPDF"""

    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """
        Extract all text from a PDF file.

        Args:
            pdf_file: File object or path to PDF

        Returns:
            str: Extracted text from the PDF
        """
        try:
            # Open PDF
            if isinstance(pdf_file, str):
                doc = fitz.open(pdf_file)
            else:
                # For uploaded files (bytes)
                doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

            text = ""
            for page in doc:
                text += page.get_text()

            doc.close()
            return text

        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return ""

# =============================================================================
# 2. TEXT PREPROCESSING & SENTENCE SPLITTING MODULE
# =============================================================================


class TextPreprocessor:
    """Clean and split text into sentences using spaCy"""

    def __init__(self):
        """Initialize spaCy model for sentence segmentation"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            st.error("spaCy model 'en_core_web_sm' not found. Please install it.")
            st.info("Run: python -m spacy download en_core_web_sm")
            self.nlp = None

    def clean_text(self, text: str) -> str:
        """
        Clean raw text by removing extra whitespace and special characters.

        Args:
            text: Raw text string

        Returns:
            str: Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using spaCy.

        Args:
            text: Input text

        Returns:
            List[str]: List of sentences
        """
        if self.nlp is None:
            return []

        # Clean text first
        text = self.clean_text(text)

        # Use spaCy for sentence segmentation
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        # Filter out very short sentences (less than 5 words)
        sentences = [s for s in sentences if len(s.split()) >= 5]

        return sentences

# =============================================================================
# 3. HYBRID FEATURE EXTRACTOR MODULE
# =============================================================================


class HybridFeatureExtractor:
    """
    Combine SBERT embeddings with lexical features (Jaccard & Edit Distance).
    This is the 'Novel Hybrid' component.
    """

    def __init__(self, model_name: str = 'paraphrase-MiniLM-L6-v2'):
        """
        Initialize SBERT model and feature extractor.

        Args:
            model_name: Name of the Sentence-BERT model
        """
        self.sbert_model = SentenceTransformer(model_name)

    def get_sbert_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Generate SBERT embeddings for sentences.

        Args:
            sentences: List of sentences

        Returns:
            np.ndarray: Embeddings matrix
        """
        embeddings = self.sbert_model.encode(
            sentences, show_progress_bar=False)
        return embeddings

    def compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            emb1, emb2: Embedding vectors

        Returns:
            float: Cosine similarity score
        """
        return cosine_similarity([emb1], [emb2])[0][0]

    def compute_jaccard_similarity(self, sent1: str, sent2: str) -> float:
        """
        Compute Jaccard similarity between two sentences.

        Args:
            sent1, sent2: Input sentences

        Returns:
            float: Jaccard similarity score
        """
        # Tokenize and convert to sets
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())

        # Compute Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)

        if len(union) == 0:
            return 0.0

        return len(intersection) / len(union)

    def compute_edit_distance_similarity(self, sent1: str, sent2: str) -> float:
        """
        Compute normalized edit distance similarity.

        Args:
            sent1, sent2: Input sentences

        Returns:
            float: Normalized edit distance similarity (0-1)
        """
        # Compute Levenshtein distance
        distance = Levenshtein.distance(sent1.lower(), sent2.lower())
        max_len = max(len(sent1), len(sent2))

        if max_len == 0:
            return 1.0

        # Normalize to similarity (1 - normalized distance)
        similarity = 1 - (distance / max_len)
        return similarity

    def extract_hybrid_features(self, sent1: str, sent2: str,
                                emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
        """
        Extract hybrid features combining SBERT, Jaccard, and Edit Distance.

        Args:
            sent1, sent2: Input sentences
            emb1, emb2: SBERT embeddings

        Returns:
            np.ndarray: Feature vector [cosine_sim, jaccard_sim, edit_sim]
        """
        # Feature 1: Cosine similarity (SBERT)
        cosine_sim = self.compute_cosine_similarity(emb1, emb2)

        # Feature 2: Jaccard similarity (Lexical)
        jaccard_sim = self.compute_jaccard_similarity(sent1, sent2)

        # Feature 3: Edit distance similarity (Lexical)
        edit_sim = self.compute_edit_distance_similarity(sent1, sent2)

        # Combine into feature vector
        features = np.array([cosine_sim, jaccard_sim, edit_sim])

        return features

# =============================================================================
# 4. PARAPHRASE CLASSIFIER MODULE
# =============================================================================


class ParaphraseClassifier:
    """
    Supervised classifier for paraphrase detection.
    Uses Logistic Regression on hybrid features.
    """

    def __init__(self):
        """Initialize classifier and scaler"""
        self.classifier = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the classifier on labeled data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,) - 1 for paraphrase, 0 for not paraphrase
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train classifier
        self.classifier.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict paraphrase labels.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Predicted labels
        """
        if not self.is_trained:
            # Use threshold-based approach if not trained
            # Consider paraphrase if cosine_sim > 0.7
            return (X[:, 0] > 0.7).astype(int)

        X_scaled = self.scaler.transform(X)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict paraphrase probabilities.

        Args:
            X: Feature matrix

        Returns:
            np.ndarray: Predicted probabilities
        """
        if not self.is_trained:
            # Use cosine similarity as probability
            probs = X[:, 0]
            return np.column_stack([1 - probs, probs])

        X_scaled = self.scaler.transform(X)
        return self.classifier.predict_proba(X_scaled)

    def save_model(self, path: str):
        """Save trained model to disk"""
        with open(path, 'wb') as f:
            pickle.dump({'classifier': self.classifier,
                        'scaler': self.scaler,
                         'is_trained': self.is_trained}, f)

    def load_model(self, path: str):
        """Load trained model from disk"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.classifier = data['classifier']
            self.scaler = data['scaler']
            self.is_trained = data['is_trained']

# =============================================================================
# 5. MAIN PARAPHRASE DETECTION ENGINE
# =============================================================================


class ParaphraseDetector:
    """Main engine combining all components"""

    def __init__(self):
        """Initialize all modules"""
        self.pdf_extractor = PDFExtractor()
        self.text_preprocessor = TextPreprocessor()
        self.feature_extractor = HybridFeatureExtractor()
        self.classifier = ParaphraseClassifier()

    def process_pdf(self, pdf_file) -> List[str]:
        """
        Process PDF and extract sentences.

        Args:
            pdf_file: Uploaded PDF file

        Returns:
            List[str]: List of sentences
        """
        # Extract text
        text = self.pdf_extractor.extract_text_from_pdf(pdf_file)

        # Split into sentences
        sentences = self.text_preprocessor.split_into_sentences(text)

        return sentences

    def detect_paraphrases(self, sentences: List[str],
                           threshold: float = 0.7) -> List[Dict]:
        """
        Detect paraphrases among sentences.

        Args:
            sentences: List of sentences to compare
            threshold: Similarity threshold for paraphrase detection

        Returns:
            List[Dict]: List of paraphrase pairs with scores
        """
        if len(sentences) < 2:
            return []

        # Generate embeddings for all sentences
        embeddings = self.feature_extractor.get_sbert_embeddings(sentences)

        # Compare all sentence pairs
        paraphrase_pairs = []

        for i, j in combinations(range(len(sentences)), 2):
            # Extract hybrid features
            features = self.feature_extractor.extract_hybrid_features(
                sentences[i], sentences[j],
                embeddings[i], embeddings[j]
            )

            # Get prediction probability
            proba = self.classifier.predict_proba(
                features.reshape(1, -1))[0][1]

            # If probability exceeds threshold, consider it a paraphrase
            if proba >= threshold:
                paraphrase_pairs.append({
                    'sentence_1': sentences[i],
                    'sentence_2': sentences[j],
                    'sentence_1_idx': i,
                    'sentence_2_idx': j,
                    'similarity_score': proba,
                    'cosine_similarity': features[0],
                    'jaccard_similarity': features[1],
                    'edit_distance_similarity': features[2]
                })

        # Sort by similarity score
        paraphrase_pairs.sort(
            key=lambda x: x['similarity_score'], reverse=True)

        return paraphrase_pairs

# =============================================================================
# 6. STREAMLIT UI
# =============================================================================


def main():
    """Main Streamlit application"""

    # Page configuration
    st.set_page_config(
        page_title="Hybrid Paraphrase Detection System",
        page_icon="📄",
        layout="wide"
    )

    # Title and description
    st.title("📄 Hybrid Paraphrase Detection System")
    st.markdown("""
    ### Novel Hybrid Approach: Siamese SBERT + Lexical Features
    
    This system combines:
    - **SBERT (Sentence-BERT)**: Deep semantic embeddings using a Siamese architecture
    - **Jaccard Similarity**: Lexical word overlap analysis
    - **Edit Distance**: Character-level similarity measurement
    - **Supervised Classification**: Logistic Regression on hybrid features
    
    Upload a PDF to detect paraphrased sentences within the document.
    """)

    # Sidebar
    st.sidebar.header("Settings")
    threshold = st.sidebar.slider(
        "Similarity Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Higher threshold = stricter paraphrase detection"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **How it works:**
    1. Upload your PDF file
    2. Text is extracted and split into sentences
    3. Hybrid features are computed for all sentence pairs
    4. Classifier identifies paraphrases
    5. Results are displayed with scores
    """)

    # File upload
    st.header("1. Upload PDF Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a single PDF document for paraphrase analysis"
    )

    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Initialize detector
            detector = ParaphraseDetector()

            # Process PDF
            sentences = detector.process_pdf(uploaded_file)

            if len(sentences) == 0:
                st.error("No sentences found in the PDF. Please check the file.")
                return

            st.success(f"✅ Extracted {len(sentences)} sentences from the PDF")

            # Show extracted sentences
            with st.expander("View Extracted Sentences"):
                for idx, sent in enumerate(sentences):
                    st.write(f"**[{idx}]** {sent}")

            # Detect paraphrases
            st.header("2. Paraphrase Detection Results")

            with st.spinner("Analyzing sentence pairs for paraphrases..."):
                paraphrase_pairs = detector.detect_paraphrases(
                    sentences, threshold)

            if len(paraphrase_pairs) == 0:
                st.warning("No paraphrases detected above the threshold.")
            else:
                st.success(
                    f"🎯 Found {len(paraphrase_pairs)} potential paraphrase pairs")

                # Display results in tabs
                tab1, tab2 = st.tabs(["📊 Summary Table", "🔍 Detailed View"])

                with tab1:
                    # Create DataFrame for summary
                    summary_data = []
                    for pair in paraphrase_pairs:
                        summary_data.append({
                            'Sentence 1 (Index)': pair['sentence_1_idx'],
                            'Sentence 2 (Index)': pair['sentence_2_idx'],
                            'Overall Score': f"{pair['similarity_score']:.3f}",
                            'SBERT Cosine': f"{pair['cosine_similarity']:.3f}",
                            'Jaccard': f"{pair['jaccard_similarity']:.3f}",
                            'Edit Distance': f"{pair['edit_distance_similarity']:.3f}"
                        })

                    df = pd.DataFrame(summary_data)
                    st.dataframe(df, use_container_width=True)

                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="paraphrase_results.csv",
                        mime="text/csv"
                    )

                with tab2:
                    # Detailed view with highlighted pairs
                    for idx, pair in enumerate(paraphrase_pairs):
                        st.markdown(f"### Paraphrase Pair #{idx + 1}")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(
                                f"**Sentence {pair['sentence_1_idx']}:**")
                            st.info(pair['sentence_1'])

                        with col2:
                            st.markdown(
                                f"**Sentence {pair['sentence_2_idx']}:**")
                            st.info(pair['sentence_2'])

                        # Metrics
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(
                            4)

                        with metric_col1:
                            st.metric("Overall Score",
                                      f"{pair['similarity_score']:.3f}")

                        with metric_col2:
                            st.metric("SBERT Cosine",
                                      f"{pair['cosine_similarity']:.3f}")

                        with metric_col3:
                            st.metric(
                                "Jaccard", f"{pair['jaccard_similarity']:.3f}")

                        with metric_col4:
                            st.metric("Edit Distance",
                                      f"{pair['edit_distance_similarity']:.3f}")

                        st.markdown("---")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Technical Stack:**
    - PyMuPDF (PDF extraction)
    - spaCy (Sentence segmentation)
    - Sentence-BERT (Embeddings)
    - Scikit-learn (Classification)
    - Streamlit (UI)
    """)


if __name__ == "__main__":
    main()
