"""
Training Module for Paraphrase Detection using Quora Question Pairs Dataset
Extracts hybrid features and trains a supervised LogisticRegression classifier
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import Levenshtein
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# HYBRID FEATURE EXTRACTOR FOR TRAINING
# =============================================================================


class QuoraTrainer:
    """
    Train a paraphrase classifier on Quora Question Pairs dataset.
    Uses hybrid features: cosine_similarity, jaccard_similarity, edit_distance_similarity
    """

    def __init__(self, quora_csv_path: str = 'quora_duplicate_questions.csv',
                 model_name: str = 'paraphrase-MiniLM-L6-v2',
                 output_dir: str = '.'):
        """
        Initialize trainer with dataset and SBERT model.

        Args:
            quora_csv_path: Path to Quora dataset CSV file
            model_name: Name of Sentence-BERT model to use
            output_dir: Directory to save trained model and metadata
        """
        self.quora_csv_path = quora_csv_path
        self.model_name = model_name
        self.output_dir = output_dir

        # Initialize SBERT model
        print(f"Loading SBERT model: {model_name}...")
        self.sbert_model = SentenceTransformer(model_name)

        # Initialize classifier and scaler
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()

        # Paths for saving model
        self.model_path = os.path.join(output_dir, 'paraphrase_classifier.pkl')
        self.metrics_path = os.path.join(output_dir, 'training_metrics.json')

        # Training metrics
        self.metrics = {}

    def compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            emb1, emb2: Embedding vectors

        Returns:
            float: Cosine similarity score
        """
        from sklearn.metrics.pairwise import cosine_similarity
        return float(cosine_similarity([emb1], [emb2])[0][0])

    def compute_jaccard_similarity(self, sent1: str, sent2: str) -> float:
        """
        Compute Jaccard similarity between two sentences.

        Args:
            sent1, sent2: Input sentences

        Returns:
            float: Jaccard similarity score
        """
        words1 = set(sent1.lower().split())
        words2 = set(sent2.lower().split())

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
        distance = Levenshtein.distance(sent1.lower(), sent2.lower())
        max_len = max(len(sent1), len(sent2))

        if max_len == 0:
            return 1.0

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
        cosine_sim = self.compute_cosine_similarity(emb1, emb2)
        jaccard_sim = self.compute_jaccard_similarity(sent1, sent2)
        edit_sim = self.compute_edit_distance_similarity(sent1, sent2)

        return np.array([cosine_sim, jaccard_sim, edit_sim])

    def load_quora_dataset(self, max_samples: int = None) -> Tuple[list, list, list]:
        """
        Load Quora Question Pairs dataset from CSV.

        Args:
            max_samples: Maximum number of samples to load (None = all)

        Returns:
            Tuple[list, list, list]: (question1_list, question2_list, label_list)
        """
        print(f"\nLoading Quora dataset from: {self.quora_csv_path}")

        if not os.path.exists(self.quora_csv_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.quora_csv_path}\n"
                "Download from: https://www.kaggle.com/quora/question-pairs-dataset"
            )

        # Load CSV
        df = pd.read_csv(self.quora_csv_path)

        # Select relevant columns
        if 'question1' not in df.columns or 'question2' not in df.columns or 'is_duplicate' not in df.columns:
            raise ValueError(
                "CSV must have columns: 'question1', 'question2', 'is_duplicate'"
            )

        # Limit samples if specified
        if max_samples:
            df = df.head(max_samples)

        # Remove rows with missing values
        df = df.dropna(subset=['question1', 'question2', 'is_duplicate'])

        q1_list = df['question1'].tolist()
        q2_list = df['question2'].tolist()
        labels = df['is_duplicate'].tolist()

        print(f"Loaded {len(q1_list)} question pairs")
        return q1_list, q2_list, labels

    def generate_training_features(self, q1_list: list, q2_list: list,
                                   labels: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate hybrid features for all question pairs.

        Args:
            q1_list: List of first questions
            q2_list: List of second questions
            labels: List of labels (1 = duplicate, 0 = not duplicate)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Feature matrix X and label vector y
        """
        print(f"\nGenerating hybrid features for {len(q1_list)} pairs...")

        X_features = []
        y_labels = []

        # Process in batches for embedding generation
        batch_size = 32
        total_pairs = len(q1_list)

        for batch_idx in range(0, total_pairs, batch_size):
            end_idx = min(batch_idx + batch_size, total_pairs)
            batch_q1 = q1_list[batch_idx:end_idx]
            batch_q2 = q2_list[batch_idx:end_idx]
            batch_labels = labels[batch_idx:end_idx]

            # Generate embeddings for both questions
            embeddings_q1 = self.sbert_model.encode(
                batch_q1, show_progress_bar=False)
            embeddings_q2 = self.sbert_model.encode(
                batch_q2, show_progress_bar=False)

            # Extract features for each pair
            for i in range(len(batch_q1)):
                features = self.extract_hybrid_features(
                    batch_q1[i], batch_q2[i],
                    embeddings_q1[i], embeddings_q2[i]
                )
                X_features.append(features)
                y_labels.append(batch_labels[i])

            # Progress indicator
            if (batch_idx + batch_size) % 128 == 0:
                print(
                    f"  Processed {min(batch_idx + batch_size, total_pairs)}/{total_pairs} pairs")

        X = np.array(X_features)
        y = np.array(y_labels)

        print(
            f"✓ Generated feature matrix: X shape = {X.shape}, y shape = {y.shape}")
        return X, y

    def train(self, X: np.ndarray, y: np.ndarray,
              test_size: float = 0.2) -> Dict:
        """
        Train the classifier on hybrid features.

        Args:
            X: Feature matrix
            y: Label vector
            test_size: Proportion of data to use for testing

        Returns:
            Dict: Training and evaluation metrics
        """
        print(f"\nTraining Logistic Regression classifier...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        print(f"  Train set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train classifier
        self.classifier.fit(X_train_scaled, y_train)

        # Predictions
        y_train_pred = self.classifier.predict(X_train_scaled)
        y_test_pred = self.classifier.predict(X_test_scaled)

        # Compute metrics
        self.metrics = {
            'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
            'test_accuracy': float(accuracy_score(y_test, y_test_pred)),
            'train_precision': float(precision_score(y_train, y_train_pred)),
            'test_precision': float(precision_score(y_test, y_test_pred)),
            'train_recall': float(recall_score(y_train, y_train_pred)),
            'test_recall': float(recall_score(y_test, y_test_pred)),
            'train_f1': float(f1_score(y_train, y_train_pred)),
            'test_f1': float(f1_score(y_test, y_test_pred)),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'feature_dimension': X.shape[1],
            'total_samples': len(X)
        }

        print(f"\n✓ Training complete!")
        print(f"  Test Accuracy:  {self.metrics['test_accuracy']:.4f}")
        print(f"  Test Precision: {self.metrics['test_precision']:.4f}")
        print(f"  Test Recall:    {self.metrics['test_recall']:.4f}")
        print(f"  Test F1-Score:  {self.metrics['test_f1']:.4f}")

        return self.metrics

    def save_model(self):
        """Save trained model and scaler to disk."""
        print(f"\nSaving model to: {self.model_path}")

        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'is_trained': True,
            'model_name': self.model_name,
            'metrics': self.metrics
        }

        os.makedirs(self.output_dir, exist_ok=True)

        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✓ Model saved successfully!")

        # Save metrics as JSON
        import json
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print(f"✓ Metrics saved to: {self.metrics_path}")

    def run_training_pipeline(self, max_samples: int = None) -> Dict:
        """
        Run complete training pipeline.

        Args:
            max_samples: Maximum number of samples to use

        Returns:
            Dict: Training metrics
        """
        print("=" * 70)
        print("PARAPHRASE DETECTION - TRAINING PIPELINE (QUORA DATASET)")
        print("=" * 70)

        # Load dataset
        q1_list, q2_list, labels = self.load_quora_dataset(max_samples)

        # Generate features
        X, y = self.generate_training_features(q1_list, q2_list, labels)

        # Train model
        metrics = self.train(X, y)

        # Save model
        self.save_model()

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)

        return metrics


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main training script execution."""

    # Check if CSV file exists in current directory
    csv_path = 'quora_duplicate_questions.csv'

    if not os.path.exists(csv_path):
        print("\n" + "!" * 70)
        print("ERROR: Quora dataset not found!")
        print("!" * 70)
        print(f"\nPlease download the dataset from:")
        print("  https://www.kaggle.com/quora/question-pairs-dataset")
        print(f"\nAnd place the 'quora_duplicate_questions.csv' file in:")
        print(f"  {os.path.abspath(csv_path)}")
        print("\n" + "!" * 70)
        sys.exit(1)

    # Initialize trainer
    trainer = QuoraTrainer(
        quora_csv_path=csv_path,
        model_name='paraphrase-MiniLM-L6-v2',
        output_dir='.'
    )

    # Run training pipeline
    # Set max_samples to None to use entire dataset, or a number to limit
    metrics = trainer.run_training_pipeline(max_samples=50000)

    print("\n✓ Training pipeline completed successfully!")
    print(f"✓ Model file: {trainer.model_path}")
    print(f"✓ Metrics file: {trainer.metrics_path}")


if __name__ == '__main__':
    main()
