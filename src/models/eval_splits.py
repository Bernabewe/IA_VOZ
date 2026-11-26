# src/models/eval_splits.py
"""
Particiones de evaluación IDÉNTICAS para los tres modelos.

  register      -> frases usadas para construir los centroides (hablantes conocidos)
  val           -> frases para calibrar umbral / Z-score / margen (hablantes conocidos)
  test          -> frases para la prueba final (hablantes conocidos)
  val_unknown   -> hablantes apartados: impostores para CALIBRAR
  test_unknown  -> otros hablantes apartados: impostores para la PRUEBA FINAL

Regla anti-fuga: se separa por FRASE dentro de cada hablante. Una frase (con sus dos
días y sus dos micrófonos) queda siempre completa en un solo bloque.
Recibe el manifiesto (DataFrame) y devuelve el mismo DataFrame con la columna 'split'.
"""
import numpy as np
import pandas as pd


def build_splits(df, n_unknown=6, n_register=9, n_val=3, seed=42):
    if n_unknown == 1 or n_unknown < 0:
        raise ValueError("n_unknown debe ser 0 o >= 2 (mitad para calibrar, mitad para probar).")

    rng = np.random.default_rng(seed)
    speakers = sorted(df["speaker"].unique())

    # Hablantes "desconocidos": no se registran; la mitad calibra y la otra mitad prueba
    unknown = [str(s) for s in rng.choice(speakers, size=n_unknown, replace=False)] if n_unknown else []
    half = n_unknown // 2
    unknown_val, unknown_test = set(unknown[:half]), set(unknown[half:])

    parts = []
    for spk in speakers:                       # orden fijo -> resultado reproducible
        rows = df[df["speaker"] == spk].copy()
        if spk in unknown_val:
            rows["split"] = "val_unknown"
        elif spk in unknown_test:
            rows["split"] = "test_unknown"
        else:
            phrases = sorted(rows["phrase"].unique())
            if len(phrases) < n_register + n_val + 1:
                raise ValueError(f"'{spk}' tiene pocas frases ({len(phrases)}) para esta partición.")
            shuffled = [phrases[i] for i in rng.permutation(len(phrases))]
            register_phrases = set(shuffled[:n_register])
            val_phrases = set(shuffled[n_register:n_register + n_val])
            rows["split"] = np.where(
                rows["phrase"].isin(register_phrases), "register",
                np.where(rows["phrase"].isin(val_phrases), "val", "test"))
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def describe_split(split_df):
    """Imprime cuántos audios y hablantes hay en cada bloque."""
    print("\nPARTICIÓN DE EVALUACIÓN (igual para los tres modelos)")
    for name, group in split_df.groupby("split"):
        print(f"   {name:<13} audios: {len(group):>4} | hablantes: {group['speaker'].nunique():>2}")
    for name in ("val_unknown", "test_unknown"):
        names = sorted(split_df.loc[split_df["split"] == name, "speaker"].unique())
        if names:
            print(f"   {name}: {', '.join(names)}")