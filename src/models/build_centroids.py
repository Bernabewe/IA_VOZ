# src/models/build_centroids.py
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

def generate_speaker_centroids(manifest_path="dataset_manifest.csv", output_db="embeddings_db/centroids.pt"):
    """
    Procesa los audios del manifiesto agrupados por hablante, muestra el progreso 
    y tiempo de ejecución por persona, y guarda los centroides L2 normalizados.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No se encontró el archivo: {manifest_path}")

    os.makedirs(os.path.dirname(output_db), exist_ok=True)
    df = pd.read_csv(manifest_path)
    
    extractor = ECAPAFeatureExtractor()
    centroids = {}
    
    # Agrupar audios por hablante
    grouped = df.groupby("speaker")
    total_speakers = len(grouped)
    
    print(f"\nIniciando extracción para {total_speakers} hablantes ({len(df)} audios totales)...")
    print("-" * 65)

    start_total_time = time.perf_counter()

    for idx, (speaker, group) in enumerate(grouped, 1):
        speaker_start_time = time.perf_counter()
        emb_list = []
        
        for file_path in group["file_path"]:
            emb = extractor.extract_embedding(file_path)
            emb_list.append(emb)

        # Calcular centroide (promedio) y normalización L2
        stacked = torch.stack(emb_list)
        mean_vector = torch.mean(stacked, dim=0)
        norm = torch.norm(mean_vector, p=2)
        if norm > 0:
            mean_vector = mean_vector / norm

        centroids[speaker] = mean_vector
        
        elapsed_speaker = time.perf_counter() - speaker_start_time
        print(f"[{idx:02d}/{total_speakers}] Hablante: '{speaker}' | Audios: {len(group):02d} | Tiempo: {elapsed_speaker:.2f} s")

    total_elapsed = time.perf_counter() - start_total_time

    # Guardar centroides
    torch.save(centroids, output_db)

    print("-" * 65)
    print("--- RESUMEN DE GENERACIÓN DE CENTROIDES ---")
    print(f"Total de hablantes procesados: {len(centroids)}")
    print(f"Tiempo total de procesamiento: {total_elapsed:.2f} s ({total_elapsed/60:.2f} min)")
    print(f"Base de datos guardada en: '{output_db}'")
    print(f"Dimensión de los centroides: {next(iter(centroids.values())).shape}")

    return centroids

if __name__ == "__main__":
    generate_speaker_centroids()