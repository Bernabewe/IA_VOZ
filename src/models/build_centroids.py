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
from src.models.model_factory import MODEL_CONFIG, DEFAULT_MODEL, create_extractor


def generate_speaker_centroids(manifest_path="dataset_manifest.csv", model_name=DEFAULT_MODEL):
    """
    Procesa los audios del manifiesto agrupados por hablante con el modelo elegido,
    y guarda los centroides L2 normalizados en la base de datos de ese modelo.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No se encontró el archivo: {manifest_path}")
    if model_name not in MODEL_CONFIG:
        raise ValueError(f"Modelo desconocido: '{model_name}'. Opciones: {list(MODEL_CONFIG)}")

    output_db = MODEL_CONFIG[model_name]["centroids"]
    os.makedirs(os.path.dirname(output_db), exist_ok=True)
    df = pd.read_csv(manifest_path)

    extractor = create_extractor(model_name)
    centroids = {}

    grouped = df.groupby("speaker")
    total_speakers = len(grouped)

    print(f"\nModelo: {model_name}")
    print(f"Iniciando extracción para {total_speakers} hablantes ({len(df)} audios totales)...")
    print("-" * 65)

    start_total_time = time.perf_counter()

    total_ok = 0
    total_omitidos = 0

    for idx, (speaker, group) in enumerate(grouped, 1):
        speaker_start_time = time.perf_counter()
        emb_list = []

        omitidos = 0
        for file_path in group["file_path"]:
            try:
                emb = extractor.extract_embedding(file_path)
                emb_list.append(emb)
            except ValueError as e:
                omitidos += 1
                print(f"   [Omitido] {file_path}: {e}")

        total_ok += len(emb_list)
        total_omitidos += omitidos

        if not emb_list:
            print(f"[{idx:02d}/{total_speakers}] Hablante '{speaker}' SIN audios válidos, se omite.")
            continue

        stacked = torch.stack(emb_list)
        mean_vector = torch.mean(stacked, dim=0)
        norm = torch.norm(mean_vector, p=2)
        if norm > 0:
            mean_vector = mean_vector / norm

        centroids[speaker] = mean_vector

        elapsed_speaker = time.perf_counter() - speaker_start_time
        print(f"[{idx:02d}/{total_speakers}] Hablante: '{speaker}' | Audios: {len(emb_list):02d}/{len(group):02d} | Omitidos: {omitidos} | Tiempo: {elapsed_speaker:.2f} s")

    total_elapsed = time.perf_counter() - start_total_time

    torch.save(centroids, output_db)

    print("-" * 65)
    print("--- RESUMEN DE GENERACIÓN DE CENTROIDES ---")
    print(f"Modelo: {model_name}")
    print(f"Total de hablantes procesados: {len(centroids)}")
    print(f"Audios procesados: {total_ok}/{len(df)} | Omitidos: {total_omitidos}")
    print(f"Tiempo total de procesamiento: {total_elapsed:.2f} s ({total_elapsed/60:.2f} min)")
    print(f"Base de datos guardada en: '{output_db}'")
    print(f"Dimensión de los centroides: {next(iter(centroids.values())).shape}")

    return centroids


if __name__ == "__main__":
    # Uso: python -m src.models.build_centroids [ECAPA-TDNN | ERes2NetV2]
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    generate_speaker_centroids(model_name=name)