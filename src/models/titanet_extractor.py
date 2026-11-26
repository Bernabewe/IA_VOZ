# src/models/titanet_extractor.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import torch
from src.data.audio_processor import load_and_normalize_audio


class TitaNetFeatureExtractor:
    """
    Encapsula TitaNet-Large (NVIDIA NeMo) para extraer embeddings de hablante
    normalizados L2. Misma interfaz que ECAPAFeatureExtractor y ERes2NetV2FeatureExtractor.

    Entrada del modelo: audio mono, 16 kHz, float32 (lo entrega load_and_normalize_audio).
    Salida: vector de embedding (la dimensión la define el modelo; no se asume).
    """
    MODEL_NAME = "titanet_large"
    SAMPLE_RATE = 16000

    def __init__(self, run_opts=None):
        print("Cargando modelo TitaNet-Large (NVIDIA NeMo)...")

        # Import diferido: si NeMo no está instalado, solo falla TitaNet (no los otros modelos)
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            raise RuntimeError(
                "No se pudo importar NeMo. Instala 'nemo-toolkit[asr]' en el entorno "
                "(Python 3.12) para usar TitaNet-Large."
            ) from e

        # Reducir los mensajes informativos de NeMo (solo errores)
        try:
            from nemo.utils import logging as nemo_logging
            nemo_logging.setLevel(40)   # 40 = ERROR
        except Exception:
            pass

        try:
            self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(self.MODEL_NAME)
        except Exception as e:
            raise RuntimeError(
                f"No se pudo cargar TitaNet-Large ('{self.MODEL_NAME}'). "
                f"Revisa la conexión a internet (primera descarga) o la instalación de NeMo: {e}"
            ) from e

        self.model.eval()   # inferencia pura (sin dither ni gradientes)
        self.device = next(self.model.parameters()).device
        self.last_inference_seconds = 0.0   # tiempo de la última inferencia (solo el modelo)

    def extract_embedding(self, file_path):
        """
        Carga un audio, lo procesa y devuelve el embedding de TitaNet normalizado L2.
        Lanza ValueError si el audio no es válido (igual que los otros extractores).
        """
        # 1. Mismo preprocesamiento compartido: mono, 16 kHz, validaciones, sin silencios
        signal, sr = load_and_normalize_audio(file_path, target_sr=self.SAMPLE_RATE)

        # 2. NeMo necesita la señal (batch, tiempo) y su longitud
        signal = signal.to(self.device)
        signal_length = torch.tensor([signal.shape[1]], dtype=torch.long, device=self.device)

        # 3. Inferencia: TitaNet calcula internamente el mel-espectrograma y devuelve (logits, embs)
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                _, embedding = self.model.forward(
                    input_signal=signal, input_signal_length=signal_length
                )
        except Exception as e:
            raise RuntimeError(f"Fallo en la inferencia de TitaNet-Large: {e}") from e
        self.last_inference_seconds = time.perf_counter() - t0

        embedding_tensor = embedding.squeeze().cpu()

        if not torch.isfinite(embedding_tensor).all():
            raise ValueError("TitaNet produjo un embedding inválido (NaN o infinito).")

        # 4. Normalización L2 unitaria (así el producto punto equivale a la similitud coseno)
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

        extractor = TitaNetFeatureExtractor()
        e1 = extractor.extract_embedding(same[0])
        e2 = extractor.extract_embedding(same[1])
        e3 = extractor.extract_embedding(other)

        print("\n--- PRUEBA TitaNet-Large ---")
        print(f"Shape: {e1.shape} | Norma L2: {torch.norm(e1, p=2).item():.4f}")
        print(f"Tiempo de inferencia (último audio): {extractor.last_inference_seconds * 1000:.1f} ms")
        print(f"Similitud MISMO hablante  ({first_speaker}): {torch.dot(e1, e2).item():.4f}")
        print(f"Similitud DISTINTO hablante: {torch.dot(e1, e3).item():.4f}")
    else:
        print("No se encontró 'dataset_manifest.csv'.")