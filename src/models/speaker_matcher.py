# src/models/speaker_matcher.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
from typing import cast
import torch
from src.models.feature_extractor import ECAPAFeatureExtractor

class SpeakerMatcher:
    """
    Carga la base de datos de centroides y realiza la identificación 1:N
    mediante producto punto matricial de similitud coseno.
    """
    def __init__(self, centroids_db_path=None, threshold=0.65):
        # Resolve centroid database path relative to project root if not provided
        if centroids_db_path is None:
            centroids_db_path = str(PROJECT_ROOT / "embeddings_db" / "centroids.pt")
        else:
            # If a relative path is given, interpret it relative to project root
            centroids_path_candidate = Path(centroids_db_path)
            if not centroids_path_candidate.is_absolute():
                centroids_db_path = str(PROJECT_ROOT / centroids_path_candidate)
        if not os.path.exists(centroids_db_path):
            raise FileNotFoundError(f"No se encontró la base de datos de centroides en: '{centroids_db_path}'")
        if not os.path.exists(centroids_db_path):
            raise FileNotFoundError(f"No se encontró la base de datos de centroides en: '{centroids_db_path}'")
        
        self.threshold = threshold
        self.extractor = ECAPAFeatureExtractor()
        
        # Cargar diccionario de centroides {speaker: tensor(192,)}
        centroids_dict = cast(dict[str, torch.Tensor], torch.load(str(centroids_db_path), weights_only=True))
        
        # Mapeo ordenado de hablantes a índice matricial
        self.speaker_names = list(centroids_dict.keys())
        
        # Construir matriz de centroides C de forma (U, 192)
        centroids_list = [centroids_dict[spk] for spk in self.speaker_names]
        self.centroids_matrix = torch.stack(centroids_list) # (U, 192)

    def identify_speaker(self, audio_file_path):
        """
        Extrae el embedding del audio de prueba y calcula las similitudes coseno.
        Retorna: (hablante_predicho, score_maximo, diccionario_scores)
        """
        # 1. Extraer embedding normalizado (192,)
        test_emb = self.extractor.extract_embedding(audio_file_path)

        # 2. Producto punto matricial -> Tensor de similitudes (U,)
        #    C (U, 192) x test_emb (192,) -> S (U,)
        scores = torch.matmul(self.centroids_matrix, test_emb)

        # 3. Obtener el máximo score e índice del hablante candidato
        max_score, max_idx = torch.max(scores, dim=0)
        max_score_val = float(max_score.item())
        candidate_speaker = self.speaker_names[int(max_idx.item())]

        # 4. Aplicar umbral adaptativo
        if max_score_val >= self.threshold:
            identified_as = candidate_speaker
        else:
            identified_as = "Desconocido"

        # Crear diccionario de puntajes ordenado
        all_scores = {spk: round(float(score.item()), 4) for spk, score in zip(self.speaker_names, scores)}

        return identified_as, max_score_val, candidate_speaker, all_scores

if __name__ == "__main__":
    import pandas as pd

    # Prueba unitaria usando una muestra del manifiesto
    if os.path.exists("dataset_manifest.csv") and os.path.exists("embeddings_db/centroids.pt"):
        df = pd.read_csv("dataset_manifest.csv")
        sample_row = df.iloc[0]
        sample_file = sample_row["file_path"]
        ground_truth = sample_row["speaker"]

        matcher = SpeakerMatcher(threshold=0.65)
        ident, score, candidate, _ = matcher.identify_speaker(sample_file)

        print("\n--- PRUEBA DE INFERENCIA BIOMÉTRICA (1:N) ---")
        print(f"Archivo probado: {sample_file}")
        print(f"Hablante real (Ground Truth): {ground_truth}")
        print(f"Candidato más cercano: {candidate} (Similitud Coseno: {score:.4f})")
        print(f"Resultado final (Umbral = 0.65): {ident}")
    else:
        print("Falta 'dataset_manifest.csv' o 'embeddings_db/centroids.pt'.")