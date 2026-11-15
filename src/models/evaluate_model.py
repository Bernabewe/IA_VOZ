# src/models/evaluate_model.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import time
import torch
import pandas as pd
from src.models.feature_extractor import ECAPAFeatureExtractor

def evaluate_out_of_sample(manifest_path="dataset_manifest.csv", train_ratio=0.8):
    """
    Divide el dataset estratificadamente por hablante (80% train / 20% test),
    construye centroides solo con train y evalúa precisión 1:N y distribuciones
    de similitud coseno en test (sin fuga de datos).
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No se encontró: {manifest_path}")

    df = pd.read_csv(manifest_path)
    extractor = ECAPAFeatureExtractor()

    train_embeddings = {}
    test_records = []

    print(f"\nDividiendo dataset ({int(train_ratio*100)}% Train / {int((1-train_ratio)*100)}% Test) por hablante...")
    
    # División determinista por hablante (semilla fija = 42)
    for speaker, group in df.groupby("speaker"):
        shuffled = group.sample(frac=1.0, random_state=42).reset_index(drop=True)
        split_idx = int(len(shuffled) * train_ratio)
        
        train_df = shuffled.iloc[:split_idx]
        test_df = shuffled.iloc[split_idx:]
        
        train_embeddings[speaker] = train_df["file_path"].tolist()
        for _, row in test_df.iterrows():
            test_records.append((speaker, row["file_path"]))

    print(f"Construyendo centroides de referencia con {sum(len(v) for v in train_embeddings.values())} audios...")
    start_time = time.perf_counter()
    
    speaker_names = sorted(list(train_embeddings.keys()))
    centroids_list = []

    for spk in speaker_names:
        embs = [extractor.extract_embedding(p) for p in train_embeddings[spk]]
        mean_vec = torch.mean(torch.stack(embs), dim=0)
        mean_vec = mean_vec / torch.norm(mean_vec, p=2)
        centroids_list.append(mean_vec)

    centroids_matrix = torch.stack(centroids_list) # (U, 192)

    print(f"Evaluando generalización sobre {len(test_records)} audios inéditos...")
    correct_top1 = 0
    genuine_scores = []
    impostor_max_scores = []

    for true_spk, file_path in test_records:
        test_emb = extractor.extract_embedding(file_path)
        scores = torch.matmul(centroids_matrix, test_emb)
        
        true_idx = speaker_names.index(true_spk)
        gen_score = scores[true_idx].item()
        genuine_scores.append(gen_score)
        
        # Máximo score entre todos los demás hablantes (impostor más cercano)
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[true_idx] = False
        max_imp_score = torch.max(scores[mask]).item()
        impostor_max_scores.append(max_imp_score)
        
        pred_idx = torch.argmax(scores).item()
        # Ensure index is an int for static type checking
        if speaker_names[int(pred_idx)] == true_spk:
            correct_top1 += 1

    elapsed = time.perf_counter() - start_time
    accuracy = (correct_top1 / len(test_records)) * 100
    
    min_gen = min(genuine_scores)
    mean_gen = sum(genuine_scores) / len(genuine_scores)
    max_imp = max(impostor_max_scores)
    mean_imp = sum(impostor_max_scores) / len(impostor_max_scores)
    suggested_threshold = (mean_gen + mean_imp) / 2

    print("\n--- REPORTE DE VALIDACIÓN ESTRICTA (OUT-OF-SAMPLE) ---")
    print(f"Tiempo total de evaluación: {elapsed:.2f} s")
    print(f"Total de pruebas inéditas evaluadas: {len(test_records)}")
    print(f"Exactitud Top-1 (Identificación Cerrada): {accuracy:.2f}% ({correct_top1}/{len(test_records)})")
    print(f"Similitud Genuina   -> Promedio: {mean_gen:.4f} | Mínima: {min_gen:.4f}")
    print(f"Similitud Impostor  -> Promedio: {mean_imp:.4f} | Máxima: {max_imp:.4f}")
    print(f"Umbral Óptimo Sugerido (Tau): {suggested_threshold:.4f}")

if __name__ == "__main__":
    evaluate_out_of_sample()