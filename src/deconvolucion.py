"""
Deconvolución ESS (método de Farina): del sweep grabado a la respuesta al impulso.

Construye el filtro inverso del sine sweep logarítmico, lo convoluciona con
la grabación y obtiene la Impulse Response (IR) del sistema acústico medido.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve


def construir_filtro_inverso(
    sweep: np.ndarray,
    f_inicio: float,
    f_fin: float,
    fs: int,
) -> np.ndarray:
    """
    Construye el filtro inverso del Exponential Sine Sweep (ESS / Farina).

    QUÉ: Devuelve una señal que, convolucionada con el sweep original,
    produce un impulso (casi un delta). Es el sweep dado vuelta en el
    tiempo, con una envolvente que compensa la energía por frecuencia.

    POR QUÉ: Un sweep logarítmico pasa más tiempo en los graves → deposita
    más energía ahí. Si solo lo invirtiéramos en el tiempo, la IR saldría
    con un sesgo espectral. La envolvente (~−6 dB/octava) lo corrige.
    Además, en el dominio temporal, las distorsiones armónicas aparecen
    ANTES del impulso lineal (ventaja Farina).

    Args:
        sweep: El mismo estímulo que se reprodujo (float, mono).
        f_inicio: Frecuencia inicial del sweep en Hz.
        f_fin: Frecuencia final del sweep en Hz.
        fs: Frecuencia de muestreo en Hz.

    Returns:
        Array float64 con el filtro inverso, normalizado para que
        sweep ⋆ filtro ≈ pico 1.
    """
    if sweep.ndim != 1:
        raise ValueError("sweep debe ser un array 1-D (mono).")
    if f_inicio <= 0 or f_fin <= f_inicio:
        raise ValueError("Se necesita 0 < f_inicio < f_fin.")
    if fs <= 0:
        raise ValueError("fs debe ser positivo.")

    n = len(sweep)
    T = n / fs
    t = np.arange(n) / fs
    L = np.log(f_fin / f_inicio)

    # Envolvente sobre el sweep invertido:
    # t=0 → agudos del original (envolvente ≈ 1, sin atenuar)
    # t=T → graves del original (envolvente ≈ f1/f2, atenuados)
    envolvente = np.exp(-t * L / T)
    filtro = sweep[::-1].astype(np.float64) * envolvente

    # Escala para que la auto-deconvolución del estímulo dé pico ~1.
    # Así la IR queda en una escala interpretable (1 ≈ eco directo unitario).
    sonda = fftconvolve(sweep.astype(np.float64), filtro, mode="full")
    pico_sonda = float(np.max(np.abs(sonda)))
    if pico_sonda < 1e-12:
        raise RuntimeError("El filtro inverso quedó nulo; revisá el sweep.")
    filtro /= pico_sonda

    return filtro


def deconvolucionar(
    grabacion: np.ndarray,
    filtro_inverso: np.ndarray,
) -> np.ndarray:
    """
    Obtiene la respuesta al impulso convolucionando grabación × filtro inverso.

    QUÉ: IR = grabacion ⋆ filtro_inverso (convolución lineal vía FFT).

    POR QUÉ: En el dominio del tiempo, "deshacer" el sweep que pasó por
    la sala+parlante+mic es equivalente a filtrar la grabación con el
    inverso del estímulo. El resultado es cómo respondería el sistema
    a un click ideal (impulso).

    Args:
        grabacion: Señal capturada por el mic (float, mono).
        filtro_inverso: Salida de construir_filtro_inverso.

    Returns:
        Array float64 con la Impulse Response completa (modo 'full').
    """
    if grabacion.ndim != 1 or filtro_inverso.ndim != 1:
        raise ValueError("grabacion y filtro_inverso deben ser 1-D (mono).")
    if len(grabacion) == 0 or len(filtro_inverso) == 0:
        raise ValueError("grabacion y filtro_inverso no pueden estar vacíos.")

    return fftconvolve(
        grabacion.astype(np.float64),
        filtro_inverso.astype(np.float64),
        mode="full",
    )


def cargar_wav_mono(ruta: Path) -> tuple[np.ndarray, int]:
    """
    Carga un WAV mono y lo devuelve en float64 normalizado a [-1, 1].

    QUÉ: Lee PCM (int16 u otro) o float y normaliza.
    POR QUÉ: Permite reutilizar resultados/grabacion.wav sin volver a
    grabar, útil para iterar sobre la deconvolución.
    """
    fs, data = wavfile.read(str(ruta))
    if data.ndim > 1:
        data = data[:, 0]

    if np.issubdtype(data.dtype, np.integer):
        # int16 → [-1, 1] usando el máximo del tipo (32767, etc.).
        max_abs = float(np.iinfo(data.dtype).max)
        senal = data.astype(np.float64) / max_abs
    else:
        senal = data.astype(np.float64)

    return senal, int(fs)


def guardar_impulso(
    impulso: np.ndarray,
    fs: int,
    ruta_wav: Path,
    ruta_npy: Path | None = None,
) -> None:
    """
    Guarda la IR como WAV (escuchable) y opcionalmente como .npy (float64).

    QUÉ: Normaliza el WAV al pico para no clipear; el .npy conserva la
    escala de deconvolución sin cuantizar a 16-bit.

    POR QUÉ: El WAV sirve para inspeccionar a oído; el .npy mantiene
    precisión numérica para el análisis FFT de la fase siguiente.
    """
    ruta_wav.parent.mkdir(parents=True, exist_ok=True)

    pico = float(np.max(np.abs(impulso)))
    if pico < 1e-12:
        raise RuntimeError("La IR está en silencio; no hay nada que guardar.")

    # WAV: normalizamos a 0.9 para dejar margen y escuchar sin clipping.
    wav_norm = (impulso / pico) * 0.9
    pcm = (np.clip(wav_norm, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(str(ruta_wav), fs, pcm)

    if ruta_npy is not None:
        ruta_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(ruta_npy), impulso.astype(np.float64))


def graficar_impulso(
    impulso: np.ndarray,
    fs: int,
    ruta: Path,
    ventana_pre_ms: float = 50.0,
    ventana_post_ms: float = 500.0,
) -> None:
    """
    Guarda un gráfico de la IR centrado en el pico principal.

    QUÉ: Recorta unos cientos de ms alrededor del máximo y dibuja
    amplitud vs tiempo.

    POR QUÉ: La convolución 'full' es muy larga; el impulso útil (eco
    directo + primeras reflexiones) está cerca del pico. Un zoom hace
    visible la forma típica de una IR de sala.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)

    idx_pico = int(np.argmax(np.abs(impulso)))
    n_pre = int(ventana_pre_ms * fs / 1000.0)
    n_post = int(ventana_post_ms * fs / 1000.0)
    i0 = max(0, idx_pico - n_pre)
    i1 = min(len(impulso), idx_pico + n_post)

    fragmento = impulso[i0:i1]
    # Eje temporal relativo al pico (0 ms = llegada del eco directo).
    t_ms = (np.arange(i0, i1) - idx_pico) / fs * 1000.0

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_ms, fragmento, linewidth=0.6, color="#1a1a1a")
    ax.axvline(0.0, color="#c0392b", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title("Impulse response (ESS deconvolution)")
    ax.set_xlabel("Time relative to peak (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(t_ms[0], t_ms[-1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
