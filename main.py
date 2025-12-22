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
from datasets import load_dataset
import json
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
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = 'paraphrase_classifier.pkl'
        self.training_cache_path = 'training_data_cache.pkl'

    def train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the classifier on features and labels.

        Args:
            X: Feature matrix
            y: Label vector
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

    def load_model(self, path: str = None):
        """Load trained model from disk"""
        if path is None:
            path = self.model_path

        if not os.path.exists(path):
            return False

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.classifier = data.get('classifier')
                self.scaler = data.get('scaler')
                self.is_trained = data.get('is_trained', False)

                # If the loaded data is incomplete, delete the corrupt file and return False
                if self.classifier is None or self.scaler is None:
                    os.remove(path)
                    print(f"Removed corrupt model file: {path}")
                    return False
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            # Remove corrupt model file
            try:
                os.remove(path)
                print(f"Removed corrupt model file: {path}")
            except:
                pass
            return False


# =============================================================================
# 5. MAIN PARAPHRASE DETECTION ENGINE
# =============================================================================


class ParaphraseDetector:
    """Main engine combining all components"""

    def __init__(self, auto_train: bool = True, use_trained_model: bool = True):
        """
        Initialize all modules.

        Args:
            auto_train: Whether to automatically load pre-trained model if it exists
            use_trained_model: Whether to use trained model (if loaded) or threshold-based approach
        """
        self.pdf_extractor = PDFExtractor()
        self.text_preprocessor = TextPreprocessor()
        self.feature_extractor = HybridFeatureExtractor()
        self.classifier = ParaphraseClassifier()
        self.use_trained_model = use_trained_model

        # Try to load pre-trained model at startup
        if auto_train:
            loaded = self.classifier.load_model()
            if loaded:
                self.training_metrics = {'loaded_from_disk': True}
            else:
                self.training_metrics = None
        else:
            self.training_metrics = None

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
                           threshold: float = 0.7,
                           use_trained: bool = None) -> List[Dict]:
        """
        Detect paraphrases among sentences.

        Args:
            sentences: List of sentences to compare
            threshold: Similarity threshold for paraphrase detection
            use_trained: Whether to use trained model (None = use instance setting)

        Returns:
            List[Dict]: List of paraphrase pairs with scores
        """
        if len(sentences) < 2:
            return []

        # Determine which model to use
        use_model = use_trained if use_trained is not None else self.use_trained_model
        use_model = use_model and self.classifier.is_trained

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
            if use_model:
                # Use trained classifier
                proba = self.classifier.predict_proba(
                    features.reshape(1, -1))[0][1]
            else:
                # Use threshold-based approach on cosine similarity
                proba = features[0]  # cosine_similarity

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

    # Model selection
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Model Selection")

    # Initialize detector first
    if 'detector' not in st.session_state:
        st.session_state.detector = ParaphraseDetector(auto_train=True)

    detector = st.session_state.detector

    # Show model status
    if detector.classifier.is_trained:
        st.sidebar.success("✅ Trained Model Available")
        use_trained = st.sidebar.checkbox("Use Trained Model", value=True,
                                          help="Use the supervised classifier (if unchecked, uses threshold-based approach)")
        detector.use_trained_model = use_trained

        if detector.training_metrics and 'loaded_from_disk' in detector.training_metrics:
            st.sidebar.info("📁 Model loaded from disk")
    else:
        st.sidebar.warning("⚠️ No trained model loaded")
        st.sidebar.info("Using threshold-based cosine similarity")
        use_trained = False

    st.sidebar.markdown("---")
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

    # Model training section
    st.sidebar.markdown("---")
    st.sidebar.header("🎓 Model Training")

    # Training button
    if st.sidebar.button("🚀 Train Model on Quora Dataset"):
        st.sidebar.error("⚠️ Training requires quora_duplicate_questions.csv\n\nPlease run: python train.py\n\n"
                         "Dataset: https://www.kaggle.com/quora/question-pairs-dataset")

    st.sidebar.markdown("""
    **To train the model:**
    1. Download Quora dataset from Kaggle
    2. Save as `quora_duplicate_questions.csv`
    3. Run: `python train.py`
    4. Reload Streamlit app
    """)

    # Main content tabs
    st.markdown("---")
    main_tabs = st.tabs(["🔍 Paraphrase Detection", "📊 Compare Two PDFs"])

    with main_tabs[0]:
        # Original paraphrase detection section
        st.header("1. Upload PDF Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a single PDF document for paraphrase analysis"
        )

    if uploaded_file is not None:
        with st.spinner("Processing PDF..."):
            # Use detector from session state
            if 'detector' not in st.session_state:
                st.session_state.detector = ParaphraseDetector(auto_train=True)

            detector = st.session_state.detector
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
                    sentences, threshold, use_trained=use_trained)

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
                    st.dataframe(df, use_container_width=True, height=400)

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

    with main_tabs[1]:
        # PDF Comparison section
        st.header("📊 Compare Two PDFs")
        st.markdown(
            "Upload two PDF documents to measure their overall similarity.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("PDF 1")
            pdf_file_1 = st.file_uploader(
                "Choose first PDF",
                type=['pdf'],
                key='pdf1',
                help="Upload first PDF for comparison"
            )

        with col2:
            st.subheader("PDF 2")
            pdf_file_2 = st.file_uploader(
                "Choose second PDF",
                type=['pdf'],
                key='pdf2',
                help="Upload second PDF for comparison"
            )

        if pdf_file_1 is not None and pdf_file_2 is not None:
            with st.spinner("Comparing PDFs..."):
                if 'detector' not in st.session_state:
                    st.session_state.detector = ParaphraseDetector(
                        auto_train=True)

                detector = st.session_state.detector

                # Extract text from both PDFs
                sentences_1 = detector.process_pdf(pdf_file_1)
                sentences_2 = detector.process_pdf(pdf_file_2)

                if len(sentences_1) == 0 or len(sentences_2) == 0:
                    st.error("Could not extract text from one or both PDFs")
                else:
                    # Calculate similarity between all sentence pairs from both PDFs
                    from sklearn.metrics.pairwise import cosine_similarity

                    # Get embeddings
                    embeddings_1 = detector.feature_extractor.get_sbert_embeddings(
                        sentences_1)
                    embeddings_2 = detector.feature_extractor.get_sbert_embeddings(
                        sentences_2)

                    # Compute similarity matrix
                    similarity_matrix = cosine_similarity(
                        embeddings_1, embeddings_2)

                    # Calculate overall similarity metrics
                    # Use average of best matches per sentence (more meaningful)
                    avg_max_per_sentence = float(
                        similarity_matrix.max(axis=1).mean())
                    max_similarity = float(similarity_matrix.max())
                    # Also calculate mean of high-scoring pairs (>0.5) for reference
                    high_scoring_pairs = similarity_matrix[similarity_matrix > 0.5]
                    if len(high_scoring_pairs) > 0:
                        mean_high_scores = float(high_scoring_pairs.mean())
                    else:
                        mean_high_scores = 0.0

                    # Use avg_max_per_sentence as primary overall similarity
                    overall_similarity = avg_max_per_sentence

                    # Display metrics
                    st.markdown("### 📈 Similarity Metrics")
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(
                        4)

                    with metric_col1:
                        st.metric(
                            "Overall Similarity",
                            f"{overall_similarity:.1%}",
                            help="Average of best matches for each sentence (best indicator)"
                        )

                    with metric_col2:
                        st.metric(
                            "Maximum Similarity",
                            f"{max_similarity:.1%}",
                            help="Highest similarity between any two sentences"
                        )

                    with metric_col3:
                        st.metric(
                            "High-Scoring Pairs (>0.5)",
                            f"{len(high_scoring_pairs)}/{similarity_matrix.size}",
                            help="Number of sentence pairs with similarity > 0.5"
                        )

                    with metric_col4:
                        st.metric(
                            "Sentences (PDF1/PDF2)",
                            f"{len(sentences_1)}/{len(sentences_2)}"
                        )

                    st.markdown("---")

                    # Show detailed comparison
                    if st.checkbox("Show Detailed Matching Pairs", value=True):
                        st.subheader("🔍 Top Matching Sentence Pairs")

                        # Get top pairs
                        top_n = st.slider(
                            "Number of top pairs to show", 1, 20, 10)

                        # Flatten and sort
                        pair_scores = []
                        for i in range(len(sentences_1)):
                            for j in range(len(sentences_2)):
                                score = float(similarity_matrix[i, j])
                                if score > 0.3:  # Only show relevant matches
                                    pair_scores.append({
                                        'pdf1_idx': i,
                                        'pdf2_idx': j,
                                        'score': score,
                                        'sentence_1': sentences_1[i][:100] + '...' if len(sentences_1[i]) > 100 else sentences_1[i],
                                        'sentence_2': sentences_2[j][:100] + '...' if len(sentences_2[j]) > 100 else sentences_2[j]
                                    })

                        # Sort by score
                        pair_scores.sort(
                            key=lambda x: x['score'], reverse=True)

                        if pair_scores:
                            # Create DataFrame
                            comparison_df = pd.DataFrame([
                                {
                                    'PDF 1 (Index)': p['pdf1_idx'],
                                    'PDF 2 (Index)': p['pdf2_idx'],
                                    'Similarity': f"{p['score']:.3f}",
                                    'PDF 1 Sentence': p['sentence_1'],
                                    'PDF 2 Sentence': p['sentence_2']
                                }
                                for p in pair_scores[:top_n]
                            ])

                            st.dataframe(
                                comparison_df, use_container_width=True, height=400)

                            # Export option
                            csv = comparison_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Comparison as CSV",
                                data=csv,
                                file_name="pdf_comparison.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info(
                                "No matching pairs found with similarity > 0.30")

                    # Show similarity heatmap info
                    st.subheader("📊 Similarity Distribution")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Similarity Score Ranges:**")
                        high_similarity = (similarity_matrix > 0.8).sum()
                        medium_similarity = ((similarity_matrix >= 0.5) & (
                            similarity_matrix <= 0.8)).sum()
                        low_similarity = (similarity_matrix < 0.5).sum()
                        total_pairs = similarity_matrix.size

                        st.text(
                            f"Very High (>0.80):  {high_similarity}/{total_pairs} pairs ({high_similarity/total_pairs*100:.1f}%)")
                        st.text(
                            f"Medium (0.50-0.80): {medium_similarity}/{total_pairs} pairs ({medium_similarity/total_pairs*100:.1f}%)")
                        st.text(
                            f"Low (<0.50):        {low_similarity}/{total_pairs} pairs ({low_similarity/total_pairs*100:.1f}%)")

                    with col2:
                        st.write("**Interpretation:**")
                        if overall_similarity > 0.75:
                            st.success(
                                "✅ Documents are highly similar (likely duplicate or very similar content)")
                        elif overall_similarity > 0.6:
                            st.info(
                                "ℹ️ Documents have significant similarity (substantial content overlap)")
                        elif overall_similarity > 0.4:
                            st.warning(
                                "⚠️ Documents have some similarity (partial overlap)")
                        else:
                            st.error(
                                "❌ Documents are largely different (minimal similarity)")

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
