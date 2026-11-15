# src/models/feature_extractor.py
import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en sys.path para ejecución directa como script
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from src.data.audio_processor import load_and_normalize_audio

class ECAPAFeatureExtractor:
    """
    Encapsula el modelo preentrenado ECAPA-TDNN de SpeechBrain
    para la extracción de embeddings de hablante de 192 dimensiones.
    """
    def __init__(self, run_opts=None):
        # 1. Cargar el modelo desde SpeechBrain / HuggingFace
        print("Cargando modelo ECAPA-TDNN (SpeechBrain)...")
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="tmp_model",
            run_opts=run_opts,
            local_strategy=LocalStrategy.COPY
        )
        if classifier is None:
            raise RuntimeError("No se pudo cargar el modelo ECAPA-TDNN desde SpeechBrain.")
        self.classifier = classifier
        # 2. Congelar gradientes para inferencia pura
        self.classifier.eval()

    def extract_embedding(self, file_path):
        """
        Carga un audio, lo procesa y extrae un embedding de 192 dimensiones normalizado L2.
        """
        # Cargar señal normalizada (1, T) a 16kHz
        signal, sr = load_and_normalize_audio(file_path, target_sr=16000)

        with torch.no_grad():
            # Inferencia: el modelo espera la señal (Batch, Time)
            embeddings = self.classifier.encode_batch(signal)
            
            # Squeeze para remover dimensiones extras -> Tensor (192,)
            embedding_tensor = embeddings.squeeze()

            # Normalización L2 unitaria
            norm = torch.norm(embedding_tensor, p=2)
            if norm > 0:
                embedding_tensor = embedding_tensor / norm

        return embedding_tensor

if __name__ == "__main__":
    import os
    import pandas as pd

    # Prueba unitaria usando el manifiesto existente
    if os.path.exists("dataset_manifest.csv"):
        df = pd.read_csv("dataset_manifest.csv")
        sample_path = df.iloc[0]["file_path"]

        extractor = ECAPAFeatureExtractor()
        emb = extractor.extract_embedding(sample_path)

        print("\n--- PRUEBA DE EXTRACCIÓN DE EMBEDDINGS ---")
        print(f"Archivo procesado: {sample_path}")
        print(f"Forma del embedding (Shape): {emb.shape}")
        print(f"Tipo de dato: {emb.dtype}")
        print(f"Norma L2 del vector: {torch.norm(emb, p=2).item():.4f}")
        print(f"Primeros 5 valores del vector: {emb[:5].tolist()}")
    else:
        print("No se encontró 'dataset_manifest.csv'. Ejecuta primero dataset_loader.py.")