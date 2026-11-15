# src/models/speaker_matcher.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import torch
from src.models.feature_extractor import ECAPAFeatureExtractor


class SpeakerMatcher:
    """
    Carga la base de datos de centroides y realiza la identificación 1:N
    combinando similitud coseno, normalización Z de cohorte (S-Norm) y margen Top-1/Top-2.
    """
    def __init__(self, centroids_db_path="embeddings_db/centroids.pt", threshold=0.59, min_z_score=2.2, min_margin=0.05):
        if not os.path.exists(centroids_db_path):
            raise FileNotFoundError(f"No se encontró la base de datos de centroides en: '{centroids_db_path}'")
        
        self.threshold = threshold
        self.min_z_score = min_z_score
        self.min_margin = min_margin
        self.extractor = ECAPAFeatureExtractor()
        
        centroids_dict = torch.load(centroids_db_path, weights_only=True)
        self.speaker_names = list(centroids_dict.keys())
        
        centroids_list = [centroids_dict[spk] for spk in self.speaker_names]
        self.centroids_matrix = torch.stack(centroids_list)  # (U, 192)

    def identify_speaker(self, audio_file_path):
        """
        Extrae el embedding del audio de prueba, calcula similitudes coseno
        y evalúa Z-Score y margen competitivo frente a la cohorte.
        """
        test_emb = self.extractor.extract_embedding(audio_file_path)

        # 1. Similitudes coseno brutas (U,)
        scores = torch.matmul(self.centroids_matrix, test_emb)

        # 2. Ordenar candidatos de mayor a menor similitud
        sorted_scores, sorted_indices = torch.sort(scores, descending=True)
        
        max_score_val = sorted_scores[0].item()
        second_score_val = sorted_scores[1].item() if len(sorted_scores) > 1 else 0.0
        candidate_speaker = self.speaker_names[sorted_indices[0].item()]
        runner_up_speaker = self.speaker_names[sorted_indices[1].item()] if len(sorted_scores) > 1 else "N/A"

        # 3. Calcular Z-Score respecto al resto de la cohorte (impostores)
        cohort_scores = sorted_scores[1:]
        cohort_mean = torch.mean(cohort_scores).item()
        cohort_std = torch.std(cohort_scores).item()
        z_score = (max_score_val - cohort_mean) / (cohort_std + 1e-6)
        margin = max_score_val - second_score_val

        # 4. Triple criterio de decisión: Umbral absoluto + Z-Score + Margen Top-1/Top-2
        if max_score_val >= self.threshold and z_score >= self.min_z_score and margin >= self.min_margin:
            identified_as = candidate_speaker
        else:
            identified_as = "Desconocido"

        details = {
            "all_scores": {spk: round(s.item(), 4) for spk, s in zip(self.speaker_names, scores)},
            "z_score": round(z_score, 2),
            "margin": round(margin, 4),
            "runner_up": runner_up_speaker,
            "runner_up_score": round(second_score_val, 4)
        }

        print(f"[Matcher] Top-1: {candidate_speaker} ({max_score_val:.4f}) | Top-2: {runner_up_speaker} ({second_score_val:.4f}) | Z-Score: {z_score:.2f} | Margen: +{margin:.4f}")

        return identified_as, max_score_val, candidate_speaker, details