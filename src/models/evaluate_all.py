# src/models/evaluate_all.py
"""
Evaluación justa de ECAPA-TDNN, ERes2NetV2 y TitaNet-Large.

Recorrido:
  1. Se construye UNA partición (eval_splits) que comparten los tres modelos.
  2. Se extraen los embeddings de cada audio (extract_embedding de cada modelo) y se guardan
     en caché junto con el tiempo que tardó cada audio.
  3. Solo se usan los audios válidos para LOS TRES modelos (mismo conjunto para todos).
  4. Centroides de cada modelo = media L2 de sus embeddings de 'register'.
  5. Similitud coseno = producto punto (embeddings con norma L2) contra esos centroides.
  6. Se calibra umbral/Z-score/margen en 'val' + 'val_unknown' y se mide en 'test' + 'test_unknown'.

Uso:
  python -m src.models.evaluate_all                       (los tres modelos, 6 desconocidos)
  python -m src.models.evaluate_all --unknown 0           (cerrada con los 30 hablantes, sin rechazo)
  python -m src.models.evaluate_all --no-cache            (recalcular embeddings)
  python -m src.models.evaluate_all --no-save-calibration (no escribir calibration.json)
  python -m src.models.evaluate_all TitaNet-Large         (solo un modelo)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import io
import json
import os
import time
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.models import metrics as M
from src.models.eval_splits import build_splits, describe_split
from src.models.model_factory import available_models, create_extractor

RESULTS_DIR = "results"
CALIBRATION_PATH = "embeddings_db/calibration.json"


# ---------------------------------------------------------------
# 1. Extracción de embeddings con caché (y tiempo por audio)
# ---------------------------------------------------------------
def extract_all(model_name, paths, use_cache=True):
    """
    Devuelve {"emb": {ruta: tensor}, "skipped": {ruta: motivo}, "secs": {ruta: segundos}}.
    'secs' mide extract_embedding completo: lectura + preprocesamiento (igual en los tres
    modelos) + inferencia. Si cambias audios o audio_processor.py, usa --no-cache.
    """
    cache_path = f"embeddings_db/eval_cache_v2_{model_name}.pt"
    cache = {"emb": {}, "skipped": {}, "secs": {}}
    if use_cache and os.path.exists(cache_path):
        cache = torch.load(cache_path, weights_only=True)

    pending = [p for p in paths if p not in cache["emb"] and p not in cache["skipped"]]
    if not pending:
        print(f"[{model_name}] Embeddings tomados de la caché.")
        return cache

    print(f"[{model_name}] Extrayendo {len(pending)} embeddings...")
    extractor = create_extractor(model_name)
    os.makedirs("embeddings_db", exist_ok=True)
    start = time.perf_counter()
    failed = 0
    for i, path in enumerate(pending, 1):
        t0 = time.perf_counter()
        try:
            with redirect_stdout(io.StringIO()):          # silenciar prints de AudioProcessor
                cache["emb"][path] = extractor.extract_embedding(path)
            cache["secs"][path] = time.perf_counter() - t0
        except ValueError as e:                          # audio rechazado por el validador
            cache["skipped"][path] = str(e)
        except Exception as e:                           # fallo inesperado: NO se guarda en caché
            failed += 1
            print(f"   [Error] {path}: {e}")
        if i % 200 == 0 or i == len(pending):
            torch.save(cache, cache_path)                # guardado parcial por si se interrumpe
            print(f"   {i}/{len(pending)} | {time.perf_counter() - start:.0f} s")
    if failed:
        print(f"[{model_name}] {failed} audios fallaron por errores inesperados (no se usarán).")
    del extractor
    return cache


# ---------------------------------------------------------------
# 2. Centroides, similitud coseno y tabla de pruebas
# ---------------------------------------------------------------
def build_centroids_from_register(register_df, emb):
    """Centroide = media de los embeddings de registro, normalizada L2 (igual que build_centroids)."""
    speakers = sorted(register_df["speaker"].unique())
    centroids = []
    for spk in speakers:
        paths = register_df.loc[register_df["speaker"] == spk, "file_path"]
        centroids.append(F.normalize(torch.stack([emb[p] for p in paths]).mean(dim=0), p=2, dim=0))
    return speakers, torch.stack(centroids)


def make_trial_table(rows, emb, centroids, speaker_idx):
    """Similitud coseno (producto punto) de cada audio contra todos los centroides."""
    embeddings = torch.stack([emb[p] for p in rows["file_path"]])
    scores = torch.matmul(embeddings, centroids.T).numpy()
    true_idx = [speaker_idx.get(s, -1) for s in rows["speaker"]]   # -1 = desconocido
    return M.make_trials(scores, true_idx, rows["file_path"], rows["speaker"],
                         rows["mic"], rows["duration_sec"])


# ---------------------------------------------------------------
# 3. Evaluación de un modelo
# ---------------------------------------------------------------
def evaluate_one_model(name, cache, split_df):
    emb = cache["emb"]
    speakers, centroids = build_centroids_from_register(split_df[split_df["split"] == "register"], emb)
    speaker_idx = {s: i for i, s in enumerate(speakers)}

    def trials_for(split_names):
        rows = split_df[split_df["split"].isin(split_names)]
        keep = rows["split"].str.endswith("_unknown") | rows["speaker"].isin(speaker_idx)
        return make_trial_table(rows[keep], emb, centroids, speaker_idx)

    val_trials = trials_for(["val", "val_unknown"])
    test_trials = trials_for(["test", "test_unknown"])

    # Calibración SOLO con validación
    if (val_trials["true_idx"] < 0).any():
        params, val_cost = M.calibrate(val_trials)
    else:
        params, val_cost = dict(M.DEFAULT_PARAMS), float("nan")
        print(f"[{name}] Sin desconocidos de validación: se usan los parámetros actuales.")

    test = M.evaluate_trials(test_trials, params)
    test_default = M.evaluate_trials(test_trials, M.DEFAULT_PARAMS)

    # Resultados por micrófono (mismos parámetros calibrados)
    by_mic = []
    for mic in sorted(set(test_trials["mics"])):
        r = M.evaluate_trials(M.subset_trials(test_trials, test_trials["mics"] == mic), params)
        by_mic.append({"modelo": name, "mic": mic, "pruebas_conocidos": r["n_known"],
                       "pruebas_desconocidos": r["n_unknown"], "acc_cerrada": r["closed_acc"],
                       "id_correcta": r["open_correct"], "frr": r["frr"], "far": r["far"],
                       "falsa_identificacion": r["false_id"], "eer": r["eer"]})

    # Una fila por prueba (para auditar y graficar)
    outcome = M.outcome_labels(test_trials, params)
    trials_df = pd.DataFrame({
        "modelo": name, "audio": test_trials["files"], "mic": test_trials["mics"],
        "duracion_s": test_trials["durations"], "hablante_real": test_trials["speakers"],
        "es_desconocido": test_trials["true_idx"] < 0,
        "hablante_top1": [speakers[i] for i in test_trials["top1_idx"]],
        "similitud": test_trials["top1"], "z_score": test_trials["z"],
        "margen": test_trials["margin"], "resultado": outcome})

    secs = np.array([cache["secs"][p] for p in split_df["file_path"] if p in cache["secs"]])
    summary = {
        "modelo": name, "hablantes_registrados": len(speakers),
        "pruebas_conocidos": test["n_known"], "pruebas_desconocidos": test["n_unknown"],
        "umbral": params["threshold"], "min_z": params["min_z_score"], "min_margen": params["min_margin"],
        "costo_validacion": val_cost,
        "acc_cerrada": test["closed_acc"], "acc_cerrada_ic_inf": test["closed_acc_ci"][0],
        "acc_cerrada_ic_sup": test["closed_acc_ci"][1],
        "id_correcta": test["open_correct"], "falsa_identificacion": test["false_id"],
        "frr": test["frr"], "frr_ic_inf": test["frr_ci"][0], "frr_ic_sup": test["frr_ci"][1],
        "far": test["far"], "far_ic_inf": test["far_ci"][0], "far_ic_sup": test["far_ci"][1],
        "eer": test["eer"], "eer_umbral": test["eer_thr"], "min_dcf": test["min_dcf"],
        "ms_media": 1000 * float(secs.mean()) if len(secs) else float("nan"),
        "ms_mediana": 1000 * float(np.median(secs)) if len(secs) else float("nan"),
        "id_correcta_defecto": test_default["open_correct"], "frr_defecto": test_default["frr"],
        "far_defecto": test_default["far"], "falsa_id_defecto": test_default["false_id"],
    }
    return summary, by_mic, trials_df, params


# ---------------------------------------------------------------
# 4. Reporte
# ---------------------------------------------------------------
def _fmt(value, scale=1.0, nd=2):
    return "n/d" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value * scale:.{nd}f}"


def print_report(summaries):
    names = [s["modelo"] for s in summaries]
    width = 16
    print("\n" + "=" * (36 + width * len(names)))
    print("COMPARACIÓN (mismas particiones, mismos audios, parámetros calibrados en validación)")
    print("=" * (36 + width * len(names)))
    print(f"{'Métrica':<36}" + "".join(f"{n:>{width}}" for n in names))
    rows = [
        ("Exactitud cerrada, sin rechazo (%)", "acc_cerrada", 100, 2),
        ("Identificación correcta, abierta (%)", "id_correcta", 100, 2),
        ("Falsa identificación (%)", "falsa_identificacion", 100, 2),
        ("FRR: falsos rechazos (%)", "frr", 100, 2),
        ("FAR: desconocidos aceptados (%)", "far", 100, 2),
        ("EER (%)", "eer", 100, 2),
        ("MinDCF (P_target=0.05)", "min_dcf", 1, 3),
        ("Tiempo medio por audio (ms)", "ms_media", 1, 1),
        ("Tiempo mediano por audio (ms)", "ms_mediana", 1, 1),
        ("Umbral calibrado", "umbral", 1, 2),
        ("Z-score mínimo calibrado", "min_z", 1, 1),
        ("Margen mínimo calibrado", "min_margen", 1, 2),
    ]
    for label, key, scale, nd in rows:
        print(f"{label:<36}" + "".join(f"{_fmt(s[key], scale, nd):>{width}}" for s in summaries))
    print("\nIntervalos de confianza de 95%:")
    for s in summaries:
        print(f"  {s['modelo']:<14} exactitud {_fmt(s['acc_cerrada'],100)}% "
              f"[{_fmt(s['acc_cerrada_ic_inf'],100,1)}-{_fmt(s['acc_cerrada_ic_sup'],100,1)}] | "
              f"FRR [{_fmt(s['frr_ic_inf'],100,1)}-{_fmt(s['frr_ic_sup'],100,1)}] | "
              f"FAR [{_fmt(s['far_ic_inf'],100,1)}-{_fmt(s['far_ic_sup'],100,1)}]")
    print("\nCon los parámetros ACTUALES (0.59 / Z 2.2 / margen 0.05), sin calibrar:")
    for s in summaries:
        print(f"  {s['modelo']:<14} id. correcta {_fmt(s['id_correcta_defecto'],100)}% | "
              f"falsa id. {_fmt(s['falsa_id_defecto'],100)}% | FRR {_fmt(s['frr_defecto'],100)}% | "
              f"FAR {_fmt(s['far_defecto'],100)}%")
    print("\nNOTA: ninguna métrica aislada define al mejor modelo. Si los intervalos de confianza "
          "de dos modelos se traslapan, la diferencia no es concluyente.")


def save_calibration(params_by_model, n_unknown, seed):
    """Escribe/actualiza embeddings_db/calibration.json (lo lee MultiModelRunner)."""
    data = {}
    if os.path.exists(CALIBRATION_PATH):
        try:
            with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    for name, p in params_by_model.items():
        data[name] = {**p, "calibrado_en": "validacion", "desconocidos": n_unknown, "semilla": seed}
    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Parámetros calibrados guardados en '{CALIBRATION_PATH}'")


# ---------------------------------------------------------------
# 5. Evaluación completa
# ---------------------------------------------------------------
def run_evaluation(manifest_path="dataset_manifest.csv", n_unknown=6, seed=42,
                   models=None, use_cache=True, write_calibration=True):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No se encontró: {manifest_path}")
    models = models or available_models()

    split_df = build_splits(pd.read_csv(manifest_path), n_unknown=n_unknown, seed=seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    split_df.to_csv(f"{RESULTS_DIR}/eval_split.csv", index=False)

    # Embeddings de cada modelo; un modelo que falla no detiene a los demás
    caches = {}
    for name in models:
        try:
            caches[name] = extract_all(name, split_df["file_path"].tolist(), use_cache=use_cache)
        except Exception as e:
            print(f"[{name}] NO se pudo evaluar: {e}")
    if not caches:
        raise RuntimeError("Ningún modelo pudo extraer embeddings.")

    # Solo audios válidos para TODOS los modelos que participan
    valid = set(split_df["file_path"])
    for cache in caches.values():
        valid &= set(cache["emb"].keys())
    omitted = len(split_df) - len(valid)
    split_df = split_df[split_df["file_path"].isin(valid)].reset_index(drop=True)
    print(f"\nAudios usados: {len(split_df)} | omitidos por el validador o por errores: {omitted}")
    describe_split(split_df)

    summaries, mic_rows, trial_tables, params_by_model = [], [], [], {}
    for name, cache in caches.items():
        summary, by_mic, trials_df, params = evaluate_one_model(name, cache, split_df)
        summaries.append(summary)
        mic_rows += by_mic
        trial_tables.append(trials_df)
        params_by_model[name] = params

    print_report(summaries)
    pd.DataFrame(summaries).to_csv(f"{RESULTS_DIR}/evaluation_summary.csv", index=False)
    pd.DataFrame(mic_rows).to_csv(f"{RESULTS_DIR}/evaluation_by_mic.csv", index=False)
    pd.concat(trial_tables).to_csv(f"{RESULTS_DIR}/evaluation_trials.csv", index=False)
    print(f"\nResultados guardados en '{RESULTS_DIR}/' (summary, by_mic, trials, eval_split)")

    if write_calibration and n_unknown > 0:
        save_calibration(params_by_model, n_unknown, seed)
    return summaries


if __name__ == "__main__":
    args = sys.argv[1:]
    n_unknown = int(args[args.index("--unknown") + 1]) if "--unknown" in args else 6
    chosen = [a for a in args if a in available_models()] or None
    run_evaluation(n_unknown=n_unknown, models=chosen, use_cache="--no-cache" not in args,
                   write_calibration="--no-save-calibration" not in args)