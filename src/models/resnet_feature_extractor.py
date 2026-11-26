
import torch
import torch.nn.functional as F

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy
from speechbrain.inference.interfaces import Pretrained

from src.data.audio_processor import load_and_normalize_audio


class ResNetFeatureExtractor:

    def __init__(self, run_opts=None):

        print("Cargando modelo ResNet...")

        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-resnet-voxceleb",
            savedir="tmp_resnet_model",
            run_opts=run_opts,
            local_strategy=LocalStrategy.COPY
        )

        self.classifier.eval()

        print("Modelo ResNet cargado correctamente.")

    def extract_embedding(self, file_path):

        # 1. Cargar y preparar el audio a 16 kHz
        signal, sr = load_and_normalize_audio(file_path)

        # 2. Extraer el embedding con ResNet
        with torch.no_grad():
            embedding = self.classifier.encode_batch(signal)

        # 3. Convertir a un vector de 256 dimensiones
        embedding = embedding.squeeze()

        # 4. Normalizar el vector
        embedding = F.normalize(embedding, p=2, dim=0)

        return embedding


if __name__ == "__main__":

    import pandas as pd

    manifest = pd.read_csv("dataset_manifest.csv")

    if manifest.empty:
        raise ValueError("El manifiesto no contiene grabaciones.")

    extractor = ResNetFeatureExtractor()

    audio_path = manifest.iloc[0]["file_path"]

    embedding = extractor.extract_embedding(audio_path)

    print("\n--- PRUEBA RESNET ---")
    print("Audio:", audio_path)
    print("Dimensiones:", embedding.shape)
    print("Tipo de dato:", embedding.dtype)
    print("Norma del embedding:", torch.norm(embedding).item())
    print("Primeros 5 valores:", embedding[:5])
