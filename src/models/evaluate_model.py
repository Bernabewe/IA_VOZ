# src/models/evaluate_model.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import io
import os
import time
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models.model_factory import MODEL_CONFIG, available_models, create_extractor


# ---------------------------------------------------------------
# 1. División train/test (idéntica para todos los modelos)
# ---------------------------------------------------------------
def split_manifest(df, train_ratio=0.8, seed=42):
    """
    División determinista por hablante (80% train / 20% test, semilla fija).
    Se hace sobre el manifiesto completo, así todos los modelos usan
    exactamente los mismos audios en train y en test.
    """
    train_rows, test_rows = [], []
    for speaker, group in df.groupby("speaker"):
        shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split_idx = int(len(shuffled) * train_ratio)
        for _, row in shuffled.iloc[:split_idx].iterrows():
            train_rows.append((speaker, row["file_path"]))
        for _, row in shuffled.iloc[split_idx:].iterrows():
            test_rows.append((speaker, row["file_path"]))
    return train_rows, test_rows


# ---------------------------------------------------------------
# 2. Extracción de embeddings con caché en disco
# ---------------------------------------------------------------
def extract_all(model_name, paths, use_cache=True):
    """
    Extrae el embedding de cada audio una sola vez y lo guarda en caché.
    Los audios que el procesador rechaza (volumen bajo, etc.) se omiten y se registran.
    Si cambias los audios o audio_processor.py, borra la carpeta embeddings_db/eval_cache_*.pt
    """
    cache_path = f"embeddings_db/eval_cache_{model_name}.pt"
    cache = {"emb": {}, "skipped": {}}
    if use_cache and os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=True)

    pending = [p for p in paths if p not in cache["emb"] and p not in cache["skipped"]]
    if pending:
        print(f"[{model_name}] Extrayendo {len(pending)} embeddings (los ya guardados se reutilizan)...")
        extractor = create_extractor(model_name)
        t0 = time.perf_counter()
        for i, path in enumerate(pending, 1):
            try:
                with redirect_stdout(io.StringIO()):   # silenciar los prints de AudioProcessor
                    cache["emb"][path] = extractor.extract_embedding(path)
            except ValueError as e:
                cache["skipped"][path] = str(e)
            if i % 100 == 0 or i == len(pending):
                elapsed = time.perf_counter() - t0
                print(f"   {i}/{len(pending)} | {elapsed:.0f} s")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache, cache_path)
        del extractor
    else:
        print(f"[{model_name}] Embeddings tomados de la caché.")

    return cache


# ---------------------------------------------------------------
# 3. Métricas
# ---------------------------------------------------------------
def compute_eer(genuine, impostor):
    """
    EER: punto donde la tasa de falsa aceptación (FAR) iguala a la de falso rechazo (FRR).
    Devuelve (eer, umbral_en_el_eer).
    """
    gen = np.sort(np.asarray(genuine))
    imp = np.sort(np.asarray(impostor))
    thresholds = np.unique(np.concatenate([gen, imp]))

    far = 1.0 - np.searchsorted(imp, thresholds, side="left") / len(imp)   # impostores aceptados
    frr = np.searchsorted(gen, thresholds, side="left") / len(gen)         # genuinos rechazados

    idx = int(np.argmin(np.abs(far - frr)))
    return float((far[idx] + frr[idx]) / 2.0), float(thresholds[idx])


def rates_at_threshold(genuine, impostor, tau):
    gen = np.asarray(genuine)
    imp = np.asarray(impostor)
    far = float((imp >= tau).mean())
    frr = float((gen < tau).mean())
    return far, frr


def evaluate_model(model_name, train_rows, test_rows, use_cache=True):
    all_paths = [p for _, p in train_rows + test_rows]
    cache = extract_all(model_name, all_paths, use_cache=use_cache)
    emb, skipped = cache["emb"], cache["skipped"]

    # Centroides construidos SOLO con train (sin fuga de datos)
    by_speaker = {}
    for spk, path in train_rows:
        if path in emb:
            by_speaker.setdefault(spk, []).append(emb[path])

    speaker_names = sorted(by_speaker)
    name_idx = {s: i for i, s in enumerate(speaker_names)}
    centroids = torch.stack([
        F.normalize(torch.stack(by_speaker[s]).mean(dim=0), p=2, dim=0)
        for s in speaker_names
    ])  # (U, 192)

    correct = 0
    n_tests = 0
    genuine, impostor = [], []

    for spk, path in test_rows:
        if path not in emb or spk not in name_idx:
            continue
        scores = torch.matmul(centroids, emb[path])
        true_idx = name_idx[spk]

        genuine.append(scores[true_idx].item())
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask[true_idx] = False
        impostor.extend(scores[mask].tolist())     # todas las pruebas "impostor"

        if int(torch.argmax(scores).item()) == true_idx:
            correct += 1
        n_tests += 1

    eer, tau_eer = compute_eer(genuine, impostor)
    cfg_tau = MODEL_CONFIG[model_name]["threshold"]
    far_cfg, frr_cfg = rates_at_threshold(genuine, impostor, cfg_tau)
    far_eer, frr_eer = rates_at_threshold(genuine, impostor, tau_eer)

    return {
        "modelo": model_name,
        "audios_omitidos": len(skipped),
        "pruebas_test": n_tests,
        "top1_%": 100.0 * correct / n_tests,
        "genuina_prom": float(np.mean(genuine)),
        "genuina_min": float(np.min(genuine)),
        "impostor_prom": float(np.mean(impostor)),
        "impostor_max": float(np.max(impostor)),
        "eer_%": 100.0 * eer,
        "umbral_eer": tau_eer,
        "far_en_eer_%": 100.0 * far_eer,
        "frr_en_eer_%": 100.0 * frr_eer,
        "umbral_actual": cfg_tau,
        "far_actual_%": 100.0 * far_cfg,
        "frr_actual_%": 100.0 * frr_cfg,
    }, list(skipped.keys())


# ---------------------------------------------------------------
# 4. Evaluación completa y comparación
# ---------------------------------------------------------------
def evaluate_and_compare(manifest_path="dataset_manifest.csv", train_ratio=0.8,
                         models=None, use_cache=True, output_csv="evaluation_comparison.csv"):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No se encontró: {manifest_path}")

    models = models or available_models()
    df = pd.read_csv(manifest_path)
    train_rows, test_rows = split_manifest(df, train_ratio=train_ratio)

    train_pct = round(train_ratio * 100)
    print(f"\nDivisión del dataset: {train_pct}% train / {100 - train_pct}% test por hablante")
    print(f"Audios en train: {len(train_rows)} | Audios en test: {len(test_rows)}")

    results = []
    skipped_files = []
    for name in models:
        print("\n" + "=" * 70)
        t0 = time.perf_counter()
        res, skipped = evaluate_model(name, train_rows, test_rows, use_cache=use_cache)
        res["tiempo_s"] = round(time.perf_counter() - t0, 1)
        results.append(res)
        skipped_files = skipped   # son los mismos para todos los modelos

    # ----- Reporte por modelo -----
    for r in results:
        print("\n" + "-" * 70)
        print(f"--- {r['modelo']} ---")
        print(f"Pruebas inéditas evaluadas: {r['pruebas_test']}")
        print(f"Exactitud Top-1 (identificación cerrada): {r['top1_%']:.2f}%")
        print(f"Similitud genuina   -> Promedio: {r['genuina_prom']:.4f} | Mínima: {r['genuina_min']:.4f}")
        print(f"Similitud impostor  -> Promedio: {r['impostor_prom']:.4f} | Máxima: {r['impostor_max']:.4f}")
        print(f"EER: {r['eer_%']:.2f}% en umbral {r['umbral_eer']:.4f}")
        print(f"Con el umbral actual ({r['umbral_actual']:.2f}): "
              f"falsa aceptación {r['far_actual_%']:.2f}% | falso rechazo {r['frr_actual_%']:.2f}%")

    # ----- Tabla comparativa -----
    print("\n" + "=" * 70)
    print("COMPARACIÓN DE MODELOS (mismos audios de train y test)")
    print("=" * 70)
    header = f"{'Métrica':<34}" + "".join(f"{r['modelo']:>18}" for r in results)
    print(header)
    print("-" * len(header))
    rows = [
        ("Exactitud Top-1 (%)", "top1_%", "{:.2f}"),
        ("EER (%)", "eer_%", "{:.2f}"),
        ("Umbral en el EER", "umbral_eer", "{:.4f}"),
        ("Similitud genuina (prom)", "genuina_prom", "{:.4f}"),
        ("Similitud impostor (prom)", "impostor_prom", "{:.4f}"),
        ("Similitud impostor (máx)", "impostor_max", "{:.4f}"),
        ("Falsa aceptación, umbral actual (%)", "far_actual_%", "{:.2f}"),
        ("Falso rechazo, umbral actual (%)", "frr_actual_%", "{:.2f}"),
    ]
    for label, key, fmt in rows:
        print(f"{label:<34}" + "".join(f"{fmt.format(r[key]):>18}" for r in results))

    best = min(results, key=lambda r: r["eer_%"])
    print(f"\nMenor EER: {best['modelo']} ({best['eer_%']:.2f}%)")
    print("Umbral sugerido por modelo (para MODEL_CONFIG en model_factory.py):")
    for r in results:
        print(f"   {r['modelo']}: {r['umbral_eer']:.2f}")

    if skipped_files:
        print(f"\nAudios omitidos por el procesador ({len(skipped_files)}), iguales para todos los modelos:")
        for p in skipped_files:
            print(f"   {p}")

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\nResultados guardados en '{output_csv}'")
    return results


if __name__ == "__main__":
    # Uso:
    #   python -m src.models.evaluate_model              (ambos modelos, con caché)
    #   python -m src.models.evaluate_model --no-cache   (recalcular todo)
    #   python -m src.models.evaluate_model ERes2NetV2   (solo un modelo)
    args = sys.argv[1:]
    use_cache = "--no-cache" not in args
    chosen = [a for a in args if a in available_models()] or None
    evaluate_and_compare(models=chosen, use_cache=use_cache)