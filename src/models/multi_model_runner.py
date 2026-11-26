# src/models/multi_model_runner.py
"""
Ejecuta ECAPA-TDNN, ERes2NetV2 y TitaNet-Large sobre LA MISMA grabación.

Recorrido de una grabación:
  1. Se valida el audio una sola vez (mismo validador que usan los tres modelos).
  2. Cada modelo, uno tras otro, hace por su cuenta:
        extractor.extract_embedding()  -> embedding del audio         (extracción de embeddings)
        centroides del modelo          -> matriz (hablantes x dim)    (centroides propios)
        producto punto                 -> similitud coseno            (embeddings con norma L2)
        umbral + Z-score + margen      -> identificado / rechazado
  3. Se mide el tiempo de cada modelo y se guarda todo en un CSV.
  4. Un error en un modelo NO detiene a los otros.

Recibe: ruta de un .wav.   Devuelve: un ComparisonResult con un ModelResult por modelo.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import csv
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from src.data.audio_processor import load_and_normalize_audio
from src.models.model_factory import available_models
from src.models.speaker_matcher import SpeakerMatcher

CALIBRATION_PATH = "embeddings_db/calibration.json"   # lo escribe la evaluación (umbral por modelo)
LOG_PATH = "results/comparison_log.csv"
DEFAULT_MIN_Z = 2.2        # mismos valores que usa hoy SpeakerMatcher
DEFAULT_MIN_MARGIN = 0.05

# Estados posibles de un modelo para una grabación
ST_IDENTIFIED = "identificado"       # superó umbral, Z-score y margen
ST_REJECTED = "rechazado"            # no superó alguno de los criterios ("Desconocido")
ST_ERROR = "error"                   # falló durante la ejecución
ST_UNAVAILABLE = "no_disponible"     # no pudo cargarse (modelo o centroides)


# ---------------------------------------------------------------
# Estructuras de resultado
# ---------------------------------------------------------------
@dataclass
class ModelResult:
    """Resultado de UN modelo sobre UNA grabación."""
    model_name: str
    status: str = ST_UNAVAILABLE
    identified_as: str = "-"          # nombre aceptado, o "Desconocido"
    candidate: str = "-"              # hablante más cercano (aunque se rechace)
    score: float = 0.0                # similitud coseno con el candidato
    z_score: float = 0.0
    margin: float = 0.0
    seconds: float = 0.0              # tiempo total de este modelo
    error: str = ""


@dataclass
class ComparisonResult:
    """Resultados de los tres modelos sobre la misma grabación."""
    audio_path: str
    audio_ok: bool = True
    audio_error: str = ""
    mic: str = ""                     # opcional, para el análisis por micrófono
    true_speaker: str = ""            # opcional, solo cuando se conoce (evaluación)
    results: Dict[str, ModelResult] = field(default_factory=dict)

    def _decided(self) -> List[ModelResult]:
        """Modelos que llegaron a una decisión (identificaron o rechazaron)."""
        return [r for r in self.results.values() if r.status in (ST_IDENTIFIED, ST_REJECTED)]

    def identified_models(self) -> List[ModelResult]:
        return [r for r in self.results.values() if r.status == ST_IDENTIFIED]

    def rejected_models(self) -> List[str]:
        return [r.model_name for r in self.results.values() if r.status == ST_REJECTED]

    def agreement(self) -> str:
        """
        'misma_persona'              -> todos los que decidieron identificaron a la misma persona
        'identificaciones_distintas' -> hay al menos dos nombres distintos
        'parcial'                    -> un solo nombre, pero algún modelo rechazó
        'todos_rechazan'             -> ningún modelo identificó a nadie
        'sin_resultados'             -> ningún modelo pudo decidir
        """
        decided = self._decided()
        if not decided:
            return "sin_resultados"
        identified = self.identified_models()
        if not identified:
            return "todos_rechazan"
        names = {r.identified_as for r in identified}
        if len(names) > 1:
            return "identificaciones_distintas"
        if len(identified) == len(decided):
            return "misma_persona"
        return "parcial"

    def highest_valid_similarity(self) -> Optional[ModelResult]:
        """
        Modelo con la mayor similitud ENTRE LOS QUE IDENTIFICARON.
        OJO: esto NO significa que sea el mejor modelo: las escalas de similitud
        de cada modelo son distintas. El rendimiento real se mide en la evaluación.
        """
        valid = self.identified_models()
        return max(valid, key=lambda r: r.score) if valid else None

    def fastest(self) -> Optional[ModelResult]:
        decided = self._decided()
        return min(decided, key=lambda r: r.seconds) if decided else None


# ---------------------------------------------------------------
# Lectura de umbrales calibrados
# ---------------------------------------------------------------
def load_calibration(path: str = CALIBRATION_PATH) -> dict:
    """
    Lee los parámetros de decisión calibrados por modelo, por ejemplo:
      {"TitaNet-Large": {"threshold": 0.62, "min_z_score": 2.0, "min_margin": 0.04}}
    Si el archivo no existe o está dañado, devuelve {} (se usan los valores actuales).
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Runner] No se pudo leer '{path}': {e}. Se usarán los umbrales por defecto.")
        return {}


# ---------------------------------------------------------------
# Ejecutor de los tres modelos
# ---------------------------------------------------------------
class MultiModelRunner:
    def __init__(self, model_names=None, calibration_path=CALIBRATION_PATH, log_path=LOG_PATH):
        self.model_names = model_names or available_models()
        self.log_path = log_path
        self.matchers: Dict[str, SpeakerMatcher] = {}
        self.load_errors: Dict[str, str] = {}

        calibration = load_calibration(calibration_path)
        for name in self.model_names:
            params = calibration.get(name, {})
            try:
                # threshold=None -> SpeakerMatcher usa el umbral provisional de MODEL_CONFIG
                self.matchers[name] = SpeakerMatcher(
                    model_name=name,
                    threshold=params.get("threshold"),
                    min_z_score=params.get("min_z_score", DEFAULT_MIN_Z),
                    min_margin=params.get("min_margin", DEFAULT_MIN_MARGIN),
                )
            except Exception as e:
                # Un modelo que no carga no impide usar los demás
                self.load_errors[name] = str(e)
                print(f"[Runner] {name} NO disponible: {e}")

    def run(self, audio_path: str, mic: str = "", true_speaker: str = "",
            save_log: bool = True) -> ComparisonResult:
        """Ejecuta los modelos, uno por uno, sobre el mismo audio."""
        comparison = ComparisonResult(audio_path=audio_path, mic=mic, true_speaker=true_speaker)

        # 1. Validar la grabación una sola vez
        try:
            load_and_normalize_audio(audio_path, target_sr=16000)
        except Exception as e:
            comparison.audio_ok = False
            comparison.audio_error = str(e)
            if save_log:
                self._append_log(comparison)
            return comparison

        # 2. Ejecución secuencial controlada (CPU: en paralelo se falsearían los tiempos)
        for name in self.model_names:
            comparison.results[name] = self._run_one(name, audio_path)

        if save_log:
            self._append_log(comparison)
        return comparison

    def _run_one(self, name: str, audio_path: str) -> ModelResult:
        """Corre un solo modelo y captura cualquier error sin propagarlo."""
        if name not in self.matchers:
            return ModelResult(model_name=name, status=ST_UNAVAILABLE,
                               error=self.load_errors.get(name, "modelo no cargado"))
        t0 = time.perf_counter()
        try:
            identified_as, score, candidate, details = self.matchers[name].identify_speaker(audio_path)
            seconds = time.perf_counter() - t0
            return ModelResult(
                model_name=name,
                status=ST_REJECTED if identified_as == "Desconocido" else ST_IDENTIFIED,
                identified_as=identified_as, candidate=candidate, score=float(score),
                z_score=float(details["z_score"]), margin=float(details["margin"]),
                seconds=seconds,
            )
        except Exception as e:
            return ModelResult(model_name=name, status=ST_ERROR,
                               seconds=time.perf_counter() - t0, error=str(e))

    def _append_log(self, comparison: ComparisonResult):
        """Agrega una fila por modelo al CSV de resultados. Un fallo aquí nunca rompe el flujo."""
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            is_new = not os.path.exists(self.log_path)
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(["fecha", "audio", "mic", "hablante_real", "audio_valido", "modelo",
                                     "estado", "identificado_como", "candidato", "similitud",
                                     "z_score", "margen", "segundos", "error"])
                now = datetime.now().isoformat(timespec="seconds")
                if not comparison.results:   # audio inválido: una sola fila informativa
                    writer.writerow([now, comparison.audio_path, comparison.mic, comparison.true_speaker,
                                     False, "-", "audio_invalido", "-", "-", "", "", "", "",
                                     comparison.audio_error])
                for r in comparison.results.values():
                    writer.writerow([now, comparison.audio_path, comparison.mic, comparison.true_speaker,
                                     True, r.model_name, r.status, r.identified_as, r.candidate,
                                     f"{r.score:.4f}", f"{r.z_score:.2f}", f"{r.margin:.4f}",
                                     f"{r.seconds:.3f}", r.error])
        except Exception as e:
            print(f"[Runner] No se pudo guardar el registro: {e}")


# ---------------------------------------------------------------
# Prueba por consola:  python -m src.models.multi_model_runner [ruta.wav]
# ---------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd

    manifest = pd.read_csv("dataset_manifest.csv") if os.path.exists("dataset_manifest.csv") else None
    audio = sys.argv[1] if len(sys.argv) > 1 else (manifest.iloc[0]["file_path"] if manifest is not None else None)
    if audio is None:
        sys.exit("Indica un .wav o ejecuta desde la carpeta que contiene dataset_manifest.csv")

    mic, real = "", ""
    if manifest is not None and (manifest["file_path"] == audio).any():
        row = manifest[manifest["file_path"] == audio].iloc[0]
        mic, real = row["mic"], row["speaker"]

    runner = MultiModelRunner()
    runner.run(audio, mic=mic, true_speaker=real, save_log=False)   # pasada de calentamiento
    res = runner.run(audio, mic=mic, true_speaker=real)             # esta sí queda en el CSV

    print("\n" + "=" * 78)
    print(f"Audio: {audio} | Hablante real: {real or '?'} | Mic: {mic or '?'}")
    if not res.audio_ok:
        print(f"Audio inválido: {res.audio_error}")
    else:
        print(f"{'Modelo':<15}{'Estado':<14}{'Hablante':<16}{'Similitud':>10}{'Tiempo (s)':>12}")
        for r in res.results.values():
            print(f"{r.model_name:<15}{r.status:<14}{r.identified_as:<16}{r.score:>10.4f}{r.seconds:>12.3f}"
                  + (f"  <- {r.error}" if r.error else ""))
        print(f"\nConcordancia: {res.agreement()}")
        best = res.highest_valid_similarity()
        fast = res.fastest()
        print(f"Mayor similitud válida: {best.model_name if best else '-'} (no implica mejor modelo)")
        print(f"Más rápido: {fast.model_name if fast else '-'}")
        print(f"Rechazaron: {res.rejected_models() or 'ninguno'}")