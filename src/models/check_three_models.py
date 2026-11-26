# src/models/check_three_models.py
# Prueba rápida: cada modelo usa SUS centroides y su umbral, sobre los mismos audios.
# OJO: los audios del manifiesto ya participaron en los centroides, así que esto
# solo comprueba que todo carga y funciona. NO mide la exactitud real.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.models.model_factory import available_models
from src.models.speaker_matcher import SpeakerMatcher


def main():
    df = pd.read_csv("dataset_manifest.csv")
    muestra = df.sample(n=3, random_state=1)   # los mismos 3 audios para los tres modelos

    # 1. Cargar un SpeakerMatcher por modelo (cada uno con sus centroides)
    matchers = {}
    for nombre in available_models():
        try:
            matchers[nombre] = SpeakerMatcher(model_name=nombre)
            print(f"[OK] {nombre}: {len(matchers[nombre].speaker_names)} hablantes, "
                  f"umbral {matchers[nombre].threshold}")
        except Exception as e:
            print(f"[ERROR] {nombre} no cargó: {e}")

    # 2. Mismo audio -> los tres modelos, sin que un error detenga a los demás
    for _, fila in muestra.iterrows():
        print("\n" + "=" * 70)
        print(f"Audio: {fila['file_path']} | Hablante real: {fila['speaker']} | {fila['mic']}")
        for nombre, matcher in matchers.items():
            try:
                identificado, puntaje, candidato, _ = matcher.identify_speaker(fila["file_path"])
                print(f"  >> {nombre}: identificado={identificado} | candidato={candidato} | "
                      f"similitud={puntaje:.4f}")
            except Exception as e:
                print(f"  >> {nombre}: ERROR -> {e}")


if __name__ == "__main__":
    main()