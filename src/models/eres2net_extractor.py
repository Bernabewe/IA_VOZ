# src/models/eres2net_extractor.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForAudioXVector, AutoFeatureExtractor
from src.data.audio_processor import load_and_normalize_audio


class ERes2NetV2FeatureExtractor:
    """
    Encapsula ERes2NetV2 (3D-Speaker) para extraer embeddings de hablante
    de 192 dimensiones normalizados L2. Misma interfaz que ECAPAFeatureExtractor.
    """
    REPO = "artyomboyko/eres2netv2-192"

    def __init__(self, run_opts=None):
        print("Cargando modelo ERes2NetV2...")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.REPO, trust_remote_code=True
        )
        self.model = AutoModelForAudioXVector.from_pretrained(
            self.REPO, trust_remote_code=True
        )
        self.model.eval()

    def extract_embedding(self, file_path):
        """
        Carga un audio, lo procesa y extrae un embedding de 192 dimensiones normalizado L2.
        """
        # Mismo preprocesamiento que ECAPA: mono, 16 kHz, sin silencios, validaciones
        signal, sr = load_and_normalize_audio(file_path, target_sr=16000)
        audio = signal.squeeze(0).numpy()

        inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            embedding_tensor = self.model(**inputs).embeddings.squeeze()

            norm = torch.norm(embedding_tensor, p=2)
            if norm > 0:
                embedding_tensor = embedding_tensor / norm

        return embedding_tensor


if __name__ == "__main__":
    import os
    import pandas as pd

    if os.path.exists("dataset_manifest.csv"):
        df = pd.read_csv("dataset_manifest.csv")
        first_speaker = df.iloc[0]["speaker"]
        same = df[df["speaker"] == first_speaker]["file_path"].tolist()[:2]
        other = df[df["speaker"] != first_speaker].iloc[0]["file_path"]

        extractor = ERes2NetV2FeatureExtractor()
        e1 = extractor.extract_embedding(same[0])
        e2 = extractor.extract_embedding(same[1])
        e3 = extractor.extract_embedding(other)

        print("\n--- PRUEBA ERes2NetV2 ---")
        print(f"Shape: {e1.shape} | Norma L2: {torch.norm(e1, p=2).item():.4f}")
        print(f"Similitud MISMO hablante  ({first_speaker}): {torch.dot(e1, e2).item():.4f}")
        print(f"Similitud DISTINTO hablante: {torch.dot(e1, e3).item():.4f}")
    else:
        print("No se encontró 'dataset_manifest.csv'.")