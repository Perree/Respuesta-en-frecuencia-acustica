"""
Generador de sine sweep logarítmico (ESS / Farina).

Este módulo crea el estímulo de medición: un tono senoidal cuya frecuencia
sube de forma exponencial en el tiempo. Más adelante (Fase 3) ese mismo
sweep permitirá obtener la respuesta al impulso por deconvolución.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def generar_sweep_logaritmico(
    f_inicio: float = 20.0,
    f_fin: float = 20000.0,
    duracion: float = 8.0,
    fs: int = 48000,
    amplitud: float = 0.5,
    fade_ms: float = 20.0,
) -> np.ndarray:
    """
    Genera un Exponential Sine Sweep (ESS) según la fórmula de Farina.

    QUÉ: Devuelve un array de muestras de un seno cuya frecuencia crece
    de forma exponencial desde f_inicio hasta f_fin.

    POR QUÉ: El sweep logarítmico es el estímulo estándar para medir la
    respuesta de un sistema acústico. Dedica el mismo tiempo relativo a
    cada octava (como el oído) y, más adelante, permite construir un
    filtro inverso que separa la respuesta lineal de las distorsiones.

    Args:
        f_inicio: Frecuencia inicial en Hz (graves).
        f_fin: Frecuencia final en Hz (agudos).
        duracion: Duración total del sweep en segundos.
        fs: Frecuencia de muestreo en Hz.
        amplitud: Amplitud pico (0–1). 0.5 deja margen contra clipping.
        fade_ms: Duración del fade de entrada/salida en milisegundos.

    Returns:
        Array float64 con el sweep normalizado a la amplitud pedida.
    """
    if f_inicio <= 0 or f_fin <= f_inicio:
        raise ValueError("Se necesita 0 < f_inicio < f_fin.")
    if duracion <= 0 or fs <= 0:
        raise ValueError("duracion y fs deben ser positivos.")
    if not 0 < amplitud <= 1:
        raise ValueError("amplitud debe estar en (0, 1].")

    n_muestras = int(duracion * fs)
    t = np.arange(n_muestras) / fs
    T = duracion

    # Fórmula ESS de Farina: la fase crece de modo que la frecuencia
    # instantánea sea f(t) = f1 * (f2/f1)^(t/T).
    # Usamos el log natural de la razón de frecuencias una sola vez.
    L = np.log(f_fin / f_inicio)
    fase = (2.0 * np.pi * f_inicio * T / L) * (np.exp(t * L / T) - 1.0)
    sweep = amplitud * np.sin(fase)

    # Fade corto en los extremos: evita clicks al reproducir el WAV
    # (salto brusco de amplitud = discontinuidad audible).
    sweep = _aplicar_fade(sweep, fs, fade_ms)

    return sweep


def _aplicar_fade(senal: np.ndarray, fs: int, fade_ms: float) -> np.ndarray:
    """
    Aplica una ventana de fade lineal al inicio y al final de la señal.

    QUÉ: Multiplica los primeros/últimos N muestras por una rampa 0→1 / 1→0.
    POR QUÉ: Sin fade, el WAV empieza y termina a media amplitud y al
    reproducirlo se oye un click. El fade es corto para no alterar casi
    nada el contenido espectral del sweep.
    """
    n_fade = int(fs * fade_ms / 1000.0)
    if n_fade <= 0:
        return senal
    if 2 * n_fade >= len(senal):
        raise ValueError("fade_ms es demasiado largo para la duración del sweep.")

    salida = senal.copy()
    rampa = np.linspace(0.0, 1.0, n_fade)
    salida[:n_fade] *= rampa
    salida[-n_fade:] *= rampa[::-1]
    return salida


def guardar_sweep_wav(sweep: np.ndarray, fs: int, ruta: Path) -> None:
    """
    Guarda el sweep como WAV PCM 16-bit.

    QUÉ: Convierte float [-1, 1] a int16 y escribe el archivo.
    POR QUÉ: WAV 16-bit es el formato más compatible para reproducir
    después por la placa de sonido de la PC (Fase 2) sin depender
    de librerías raras.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    # Clip por seguridad: si amplitud > 1 por error, no corrompemos el WAV.
    sweep_clip = np.clip(sweep, -1.0, 1.0)
    pcm = (sweep_clip * 32767.0).astype(np.int16)
    wavfile.write(str(ruta), fs, pcm)


def graficar_waveform(sweep: np.ndarray, fs: int, ruta: Path) -> None:
    """
    Guarda un gráfico de la forma de onda del sweep.

    QUÉ: Dibuja amplitud vs tiempo y lo guarda en PNG.
    POR QUÉ: Sirve para verificar a ojo que el sweep se densifica con el
    tiempo (las oscilaciones se juntan a medida que sube la frecuencia).
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(sweep)) / fs

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, sweep, linewidth=0.4, color="#1a1a1a")
    ax.set_title("Logarithmic sine sweep (ESS)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
