# src/models/metrics.py
"""
Métricas de evaluación (solo numpy; no sabe nada de modelos ni de la interfaz).

Una "tabla de pruebas" (trials) guarda, por cada audio de prueba:
  scores   -> similitud coseno contra cada centroide registrado   (N x U)
  top1, z, margin, top1_idx -> lo que usa la regla de decisión de SpeakerMatcher
  true_idx -> índice del hablante real (-1 si es un DESCONOCIDO)

Regla de decisión (la misma de SpeakerMatcher): se acepta si top1 >= umbral
Y z-score >= min_z_score Y margen >= min_margin.

Resultados posibles de cada prueba:
  correcta             conocido, aceptado y con el nombre correcto
  falsa_identificacion conocido, aceptado pero con el nombre de OTRA persona
  falso_rechazo        conocido, rechazado
  falsa_aceptacion     desconocido, aceptado        (la base de FAR)
  rechazo_correcto     desconocido, rechazado
"""
import numpy as np

TAU_GRID = np.round(np.arange(0.20, 0.96, 0.01), 2)
Z_GRID = [0.0, 1.0, 1.5, 2.0, 2.2, 2.5, 3.0]
MARGIN_GRID = [0.0, 0.02, 0.05, 0.08, 0.10]
DEFAULT_PARAMS = {"threshold": 0.59, "min_z_score": 2.2, "min_margin": 0.05}  # valores actuales


# ---------------------------------------------------------------
# Utilidades estadísticas
# ---------------------------------------------------------------
def rate(k, n):
    return float(k) / n if n > 0 else float("nan")


def wilson_interval(k, n, z=1.96):
    """Intervalo de confianza de 95% para una proporción k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(centre - half), float(centre + half))


# ---------------------------------------------------------------
# Tabla de pruebas
# ---------------------------------------------------------------
def make_trials(scores, true_idx, files, speakers, mics, durations):
    """Construye la tabla de pruebas a partir de la matriz de similitudes (N x U)."""
    scores = np.asarray(scores, dtype=float)
    ordered = -np.sort(-scores, axis=1)            # de mayor a menor
    top1, second, cohort = ordered[:, 0], ordered[:, 1], ordered[:, 1:]
    # Mismo cálculo que SpeakerMatcher (desviación estándar muestral, ddof=1)
    z = (top1 - cohort.mean(axis=1)) / (cohort.std(axis=1, ddof=1) + 1e-6)
    return {
        "scores": scores, "top1": top1, "top1_idx": scores.argmax(axis=1),
        "z": z, "margin": top1 - second, "true_idx": np.asarray(true_idx, dtype=int),
        "files": np.asarray(files), "speakers": np.asarray(speakers),
        "mics": np.asarray(mics), "durations": np.asarray(durations, dtype=float),
    }


def subset_trials(trials, mask):
    return {key: value[mask] for key, value in trials.items()}


def decision_flags(trials, threshold, min_z_score, min_margin):
    """Aplica la regla de decisión y clasifica cada prueba (arreglos booleanos)."""
    accepted = (trials["top1"] >= threshold) & (trials["z"] >= min_z_score) & (trials["margin"] >= min_margin)
    known = trials["true_idx"] >= 0
    top_ok = trials["top1_idx"] == trials["true_idx"]
    return {
        "known": known,
        "correct": known & accepted & top_ok,
        "false_id": known & accepted & ~top_ok,
        "false_reject": known & ~accepted,
        "false_accept": ~known & accepted,
        "correct_reject": ~known & ~accepted,
    }


def outcome_labels(trials, params):
    f = decision_flags(trials, params["threshold"], params["min_z_score"], params["min_margin"])
    return np.select(
        [f["correct"], f["false_id"], f["false_reject"], f["false_accept"], f["correct_reject"]],
        ["correcta", "falsa_identificacion", "falso_rechazo", "falsa_aceptacion", "rechazo_correcto"],
        default="?")


# ---------------------------------------------------------------
# EER y MinDCF (sobre puntajes, sin Z-score ni margen)
# ---------------------------------------------------------------
def _error_curves(genuine, impostor):
    gen, imp = np.sort(np.asarray(genuine, float)), np.sort(np.asarray(impostor, float))
    thr = np.unique(np.concatenate([gen, imp]))
    thr = np.append(thr, thr[-1] + 1e-6)                       # punto "no aceptar a nadie"
    far = 1.0 - np.searchsorted(imp, thr, side="left") / len(imp)   # impostores aceptados
    frr = np.searchsorted(gen, thr, side="left") / len(gen)         # genuinos rechazados
    return thr, far, frr


def compute_eer(genuine, impostor):
    """EER: punto donde FAR = FRR. Devuelve (eer, umbral)."""
    thr, far, frr = _error_curves(genuine, impostor)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2.0), float(thr[i])


def compute_min_dcf(genuine, impostor, p_target=0.05, c_miss=1.0, c_fa=1.0):
    """MinDCF normalizado (1.0 = no mejor que un sistema sin información)."""
    _, far, frr = _error_curves(genuine, impostor)
    dcf = c_miss * frr * p_target + c_fa * far * (1.0 - p_target)
    return float(dcf.min() / min(c_miss * p_target, c_fa * (1.0 - p_target)))


def score_distributions(trials):
    """Puntajes genuinos (contra el centroide real) e impostores (contra los demás)."""
    S, known, true_idx = trials["scores"], trials["true_idx"] >= 0, trials["true_idx"]
    rows = np.where(known)[0]
    genuine = S[rows, true_idx[rows]]
    mask = np.ones((len(rows), S.shape[1]), dtype=bool)
    mask[np.arange(len(rows)), true_idx[rows]] = False
    impostor = np.concatenate([S[rows][mask], S[~known].ravel()])
    return genuine, impostor


# ---------------------------------------------------------------
# Evaluación y calibración
# ---------------------------------------------------------------
def evaluate_trials(trials, params):
    """Todas las métricas para un conjunto de pruebas y unos parámetros de decisión."""
    f = decision_flags(trials, params["threshold"], params["min_z_score"], params["min_margin"])
    n_known, n_unknown = int(f["known"].sum()), int((~f["known"]).sum())
    closed_ok = int((trials["top1_idx"][f["known"]] == trials["true_idx"][f["known"]]).sum())
    n_frr, n_far = int(f["false_reject"].sum()), int(f["false_accept"].sum())

    genuine, impostor = score_distributions(trials)
    have_scores = len(genuine) > 0 and len(impostor) > 0
    eer, eer_thr = compute_eer(genuine, impostor) if have_scores else (float("nan"), float("nan"))

    return {
        "n_known": n_known, "n_unknown": n_unknown,
        "closed_acc": rate(closed_ok, n_known), "closed_acc_ci": wilson_interval(closed_ok, n_known),
        "open_correct": rate(f["correct"].sum(), n_known),
        "false_id": rate(f["false_id"].sum(), n_known),
        "frr": rate(n_frr, n_known), "frr_ci": wilson_interval(n_frr, n_known),
        "far": rate(n_far, n_unknown), "far_ci": wilson_interval(n_far, n_unknown),
        "eer": eer, "eer_thr": eer_thr,
        "min_dcf": compute_min_dcf(genuine, impostor) if have_scores else float("nan"),
    }


def calibrate(trials):
    """
    Busca (umbral, Z-score, margen) que minimiza el error balanceado en VALIDACIÓN:
        0.5 * FAR_desconocidos + 0.5 * (FRR + falsa_identificación)_conocidos
    Devuelve (parámetros, costo). Si no hay desconocidos, el costo solo usa conocidos.
    """
    known = trials["true_idx"] >= 0
    n_known, n_unknown = int(known.sum()), int((~known).sum())
    best_cost, best_params = None, None
    for tau in TAU_GRID:
        for z in Z_GRID:
            for margin in MARGIN_GRID:
                f = decision_flags(trials, tau, z, margin)
                known_err = (f["false_reject"].sum() + f["false_id"].sum()) / n_known
                if n_unknown:
                    cost = 0.5 * (known_err + f["false_accept"].sum() / n_unknown)
                else:
                    cost = known_err
                if best_cost is None or cost < best_cost - 1e-12:
                    best_cost = cost
                    best_params = {"threshold": float(tau), "min_z_score": float(z),
                                   "min_margin": float(margin)}
    return best_params, float(best_cost)