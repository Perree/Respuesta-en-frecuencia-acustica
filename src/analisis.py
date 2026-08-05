"""
Análisis de la respuesta al impulso: FFT y RT60 opcional.

Toma la Impulse Response (IR) de la deconvolución ESS y genera:
1) Respuesta en frecuencia (magnitud en dB vs frecuencia).
2) Opcionalmente, tiempo de reverberación RT60 por integración de Schroeder.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def calcular_respuesta_frecuencia(
    impulso: np.ndarray,
    fs: int,
    f_min: float = 20.0,
    f_max: float | None = None,
    post_pico_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la magnitud espectral de la IR en dB (respuesta en frecuencia).

    QUÉ: Recorta un tramo útil alrededor del pico de la IR, aplica una
    ventana Hann, hace FFT real (rfft) y convierte |H(f)| a dB relativos
    al máximo (0 dB = pico espectral).

    POR QUÉ: La IR completa de la convolución 'full' es muy larga y tiene
    ruido / cola inútil. El pico es el sonido directo; lo que sigue son
    reflexiones. Ese tramo es lo que caracteriza cómo el sistema enfatiza
    o atenúa cada frecuencia. Escala log en Hz (más abajo, en el gráfico)
    porque el oído percibe octavas, no hertz lineales.

    Args:
        impulso: Respuesta al impulso (float, mono).
        fs: Frecuencia de muestreo en Hz.
        f_min: Frecuencia mínima a devolver (Hz).
        f_max: Frecuencia máxima (Hz). None → Nyquist.
        post_pico_s: Segundos de IR a tomar después del pico para la FFT.

    Returns:
        (frecuencias_Hz, magnitud_dB) ya recortados a [f_min, f_max].
    """
    if impulso.ndim != 1:
        raise ValueError("impulso debe ser un array 1-D (mono).")
    if fs <= 0:
        raise ValueError("fs debe ser positivo.")
    if len(impulso) < 8:
        raise ValueError("La IR es demasiado corta para analizar.")

    if f_max is None:
        f_max = fs / 2.0
    f_max = min(float(f_max), fs / 2.0)
    if not (0 < f_min < f_max):
        raise ValueError("Se necesita 0 < f_min < f_max <= Nyquist.")

    fragmento = _recortar_desde_pico(impulso, fs, post_pico_s=post_pico_s)

    # Hann: suaviza bordes del recorte → menos leakage espectral en la FFT.
    ventana = np.hanning(len(fragmento))
    espectro = np.fft.rfft(fragmento * ventana)
    frecuencias = np.fft.rfftfreq(len(fragmento), d=1.0 / fs)
    magnitud = np.abs(espectro)

    mascara = (frecuencias >= f_min) & (frecuencias <= f_max)
    frecuencias = frecuencias[mascara]
    magnitud = magnitud[mascara]

    pico = float(np.max(magnitud))
    if pico < 1e-18:
        raise RuntimeError("Espectro nulo; revisá la IR.")

    # 0 dB = componente más fuerte del tramo analizado (forma relativa).
    magnitud_db = 20.0 * np.log10(magnitud / pico + 1e-18)
    return frecuencias, magnitud_db


def graficar_respuesta_frecuencia(
    frecuencias: np.ndarray,
    magnitud_db: np.ndarray,
    ruta: Path,
) -> None:
    """
    Guarda el gráfico de respuesta en frecuencia (dB vs Hz, eje X log).

    QUÉ: Curva de magnitud relativa en dB con frecuencias en escala log.
    POR QUÉ: Así se ve de un vistazo si el sistema es “plano”, si hay
    un bache en medios, un pico de resonancia, etc. — lectura típica
    de un Bode de magnitud acústico.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.semilogx(frecuencias, magnitud_db, linewidth=1.0, color="#1a1a1a")
    ax.set_title("Frequency response (FFT of impulse response)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB, relative to peak)")
    ax.set_xlim(frecuencias[0], frecuencias[-1])
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def calcular_rt60(
    impulso: np.ndarray,
    fs: int,
    metodo: str = "T30",
    post_pico_s: float = 2.0,
) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """
    Estima RT60 a partir de la IR con integración de Schroeder (EDC).

    QUÉ: Integra hacia atrás la energía h²(t) → curva de decaimiento en dB.
    Encaja una recta en un tramo útil (T20 o T30) y extrapola a −60 dB.

    POR QUÉ: Medir −60 dB reales casi nunca se puede (el piso de ruido
    del mic lo tapa). Por eso se mide un tramo más corto (−5 a −25/−35 dB)
    y se proyecta linealmente. RT60 ≈ cuánto tarda el nivel a caer 60 dB
    después de apagar la fuente: métrica clásica de reverberación de sala.

    Args:
        impulso: Respuesta al impulso (float, mono).
        fs: Frecuencia de muestreo en Hz.
        metodo: "T20" (−5…−25 dB) o "T30" (−5…−35 dB).
        post_pico_s: Segundos después del pico usados para la EDC.

    Returns:
        (rt60_s, tiempos_s, edc_db, info) donde info trae pendiente,
        tramo de ajuste y el RT intermedio (T20/T30) para depurar.
    """
    if impulso.ndim != 1:
        raise ValueError("impulso debe ser un array 1-D (mono).")
    if fs <= 0:
        raise ValueError("fs debe ser positivo.")
    if metodo not in ("T20", "T30"):
        raise ValueError('metodo debe ser "T20" o "T30".')

    # Tramos estándar ISO 3382 (simplificados, sin filtrado por octavas).
    if metodo == "T20":
        db_hi, db_lo, span_db = -5.0, -25.0, 20.0
    else:
        db_hi, db_lo, span_db = -5.0, -35.0, 30.0

    cola = _recortar_desde_pico(impulso, fs, post_pico_s=post_pico_s)
    energia = cola.astype(np.float64) ** 2

    # Integración de Schroeder: E(t) = ∫_t^∞ h²(τ) dτ
    # cumsum al revés evita un bucle lento y es numéricamente estable.
    edc = np.cumsum(energia[::-1])[::-1]
    edc_max = float(np.max(edc))
    if edc_max < 1e-24:
        raise RuntimeError("Energía nula en la IR; no se puede estimar RT60.")

    edc_db = 10.0 * np.log10(edc / edc_max + 1e-18)
    tiempos = np.arange(len(edc_db)) / fs

    mascara = (edc_db <= db_hi) & (edc_db >= db_lo)
    if np.count_nonzero(mascara) < 8:
        raise RuntimeError(
            f"No hay tramo suficiente para {metodo}: el SNR de la IR "
            "no alcanza el rango de decaimiento pedido. Probá T20, "
            "subí el volumen del parlante o acortá post_pico_s."
        )

    t_fit = tiempos[mascara]
    db_fit = edc_db[mascara]

    # Ajuste lineal: dB(t) ≈ m*t + b  (m < 0 en un decaimiento real).
    pendiente, ordenada = np.polyfit(t_fit, db_fit, 1)
    if pendiente >= 0:
        raise RuntimeError(
            "La EDC no decae en el tramo de ajuste (pendiente ≥ 0). "
            "Revisá la IR / el ruido de fondo."
        )

    # Tiempo para caer `span_db` en la recta, escalado a 60 dB.
    t_span = span_db / (-pendiente)
    rt60 = t_span * (60.0 / span_db)

    info = {
        "metodo": metodo,
        "pendiente_db_por_s": float(pendiente),
        "ordenada_db": float(ordenada),
        "t_inicio_s": float(t_fit[0]),
        "t_fin_s": float(t_fit[-1]),
        "rt_intermedio_s": float(t_span),
    }
    return float(rt60), tiempos, edc_db, info


def graficar_decaimiento(
    tiempos: np.ndarray,
    edc_db: np.ndarray,
    info: dict,
    rt60_s: float,
    ruta: Path,
) -> None:
    """
    Guarda el gráfico de la curva de decaimiento de Schroeder + ajuste.

    QUÉ: EDC en dB vs tiempo, con la recta de T20/T30 y el RT60 estimado.
    POR QUÉ: En entrevista/demostración conviene mostrar la curva, no
    solo el número: se ve si el ajuste es razonable o si el ruido lo tuerce.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)

    pendiente = info["pendiente_db_por_s"]
    ordenada = info["ordenada_db"]
    t0 = info["t_inicio_s"]
    t1 = info["t_fin_s"]
    metodo = info["metodo"]

    # Misma recta del polyfit, dibujada un poco más allá del tramo.
    t_recta = np.linspace(max(0.0, t0 - 0.05), t1 + 0.1, 200)
    recta_db = pendiente * t_recta + ordenada

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(tiempos, edc_db, linewidth=1.0, color="#1a1a1a", label="Schroeder EDC")
    ax.plot(
        t_recta,
        recta_db,
        linewidth=1.5,
        color="#c0392b",
        linestyle="--",
        label=f"{metodo} fit → RT60 = {rt60_s:.2f} s",
    )
    ax.axvspan(t0, t1, color="#c0392b", alpha=0.08, label=f"{metodo} window")
    ax.set_title("Energy decay curve (Schroeder) and RT60 estimate")
    ax.set_xlabel("Time after IR peak (s)")
    ax.set_ylabel("Energy (dB, relative to max)")
    ax.set_ylim(-80, 5)
    ax.set_xlim(0.0, min(float(tiempos[-1]), max(rt60_s * 1.1, t1 + 0.2)))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def _recortar_desde_pico(
    impulso: np.ndarray,
    fs: int,
    post_pico_s: float,
    pre_pico_ms: float = 5.0,
) -> np.ndarray:
    """
    Extrae el tramo de IR desde un poco antes del pico hasta post_pico_s.

    QUÉ: Localiza el máximo absoluto y corta [pico−pre, pico+post].
    POR QUÉ: El pico ≈ sonido directo. Antes suele haber ruido/artefactos
    de la deconvolución; después está la cola reverberante que importa
    para FFT y RT60. Trabajar sobre ese recorte evita diluir la energía
    útil con cientos de ms de silencio/ruido.
    """
    idx_pico = int(np.argmax(np.abs(impulso)))
    n_pre = int(pre_pico_ms * fs / 1000.0)
    n_post = int(post_pico_s * fs)
    i0 = max(0, idx_pico - n_pre)
    i1 = min(len(impulso), idx_pico + n_post)
    fragmento = impulso[i0:i1].astype(np.float64)

    if len(fragmento) < 8:
        raise ValueError("Recorte de IR demasiado corto; aumentá post_pico_s.")
    return fragmento
