"""
Reproducción y grabación simultánea del sweep en la PC (sounddevice).

Este módulo envía el sweep por los parlantes/salida de audio y, al mismo
tiempo, captura lo que llega al micrófono. El WAV resultante es la
respuesta del sistema acústico (sala + parlante + mic) al estímulo ESS.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

try:
    import sounddevice as sd
except ImportError as exc:
    raise ImportError(
        "No se encontró el paquete 'sounddevice'. "
        "Instalalo con: pip install sounddevice"
    ) from exc


def listar_dispositivos() -> None:
    """
    Imprime la lista de dispositivos de audio que PortAudio ve en el sistema.

    QUÉ: Llama a sd.query_devices() y muestra índice, nombre y canales I/O.
    POR QUÉ: En Windows hay varios dispositivos (Realtek, auriculares USB,
    virtuales de Zoom, etc.). Sin esta lista no sabés qué índice pasar
    a reproducir_y_grabar si el default no es el correcto.
    """
    try:
        dispositivos = sd.query_devices()
    except Exception as exc:
        raise RuntimeError(
            "No se pudo consultar los dispositivos de audio. "
            "¿Está PortAudio instalado / la placa de sonido activa? "
            f"Detalle: {exc}"
        ) from exc

    print("Available audio devices (sounddevice / PortAudio):")
    print(dispositivos)
    print()
    try:
        default_in, default_out = sd.default.device
        print(f"  Default input  : {default_in}")
        print(f"  Default output : {default_out}")
    except Exception:
        print("  (Could not read default devices)")
    print()


def _validar_dispositivos(
    dispositivo_entrada: int | None,
    dispositivo_salida: int | None,
) -> None:
    """
    Comprueba que existan entrada (mic) y salida (parlante) usables.

    QUÉ: Verifica defaults o índices pedidos contra sd.query_devices().
    POR QUÉ: Fallar acá con un mensaje claro evita un PortAudioError
    críptico a mitad de la grabación.
    """
    try:
        dispositivos = sd.query_devices()
    except Exception as exc:
        raise RuntimeError(
            "PortAudio no pudo enumerar dispositivos. "
            f"Detalle: {exc}"
        ) from exc

    if len(dispositivos) == 0:
        raise RuntimeError(
            "No hay dispositivos de audio. Revisá que la placa de sonido "
            "esté habilitada en Windows."
        )

    # None = dejar que sounddevice use el default del sistema.
    idx_in = (
        dispositivo_entrada
        if dispositivo_entrada is not None
        else sd.default.device[0]
    )
    idx_out = (
        dispositivo_salida
        if dispositivo_salida is not None
        else sd.default.device[1]
    )

    if idx_in is None or idx_in < 0:
        raise RuntimeError(
            "No hay micrófono / dispositivo de entrada por defecto. "
            "Pasá dispositivo_entrada=N con un índice de listar_dispositivos()."
        )
    if idx_out is None or idx_out < 0:
        raise RuntimeError(
            "No hay parlante / dispositivo de salida por defecto. "
            "Pasá dispositivo_salida=N con un índice de listar_dispositivos()."
        )

    try:
        info_in = sd.query_devices(idx_in, "input")
        info_out = sd.query_devices(idx_out, "output")
    except Exception as exc:
        raise RuntimeError(
            f"Índice de dispositivo inválido (in={idx_in}, out={idx_out}). "
            "Corré listar_dispositivos() y elegí índices válidos. "
            f"Detalle: {exc}"
        ) from exc

    if info_in["max_input_channels"] < 1:
        raise RuntimeError(
            f"El dispositivo {idx_in} ('{info_in['name']}') no tiene "
            "canales de entrada (no es un micrófono)."
        )
    if info_out["max_output_channels"] < 1:
        raise RuntimeError(
            f"El dispositivo {idx_out} ('{info_out['name']}') no tiene "
            "canales de salida (no es un parlante)."
        )


def reproducir_y_grabar(
    sweep: np.ndarray,
    fs: int,
    dispositivo_entrada: int | None = None,
    dispositivo_salida: int | None = None,
    cola_s: float = 1.5,
) -> np.ndarray:
    """
    Reproduce el sweep y graba el micrófono al mismo tiempo.

    QUÉ: Usa sd.playrec (dúplex): manda muestras a la salida mientras
    lee muestras de la entrada, con el mismo reloj de muestreo.

    POR QUÉ: En una medición ESS la reproducción y la captura deben
    estar sincronizadas. Si grabás aparte, no sabés el alineamiento
    temporal entre estímulo y respuesta. playrec lo resuelve en un solo
    stream PortAudio.

    Args:
        sweep: Señal a reproducir (float, mono).
        fs: Frecuencia de muestreo en Hz (debe coincidir con el sweep).
        dispositivo_entrada: Índice del mic (None = default del sistema).
        dispositivo_salida: Índice del parlante (None = default).
        cola_s: Segundos de silencio al final del playback. Sirve para
            capturar la cola reverberante de la sala (útil en Fase 3+).

    Returns:
        Array float64 mono con la grabación (misma fs).
    """
    if sweep.ndim != 1:
        raise ValueError("sweep debe ser un array 1-D (mono).")
    if fs <= 0:
        raise ValueError("fs debe ser positivo.")
    if cola_s < 0:
        raise ValueError("cola_s no puede ser negativo.")

    _validar_dispositivos(dispositivo_entrada, dispositivo_salida)

    # Padding de silencio: el mic sigue grabando después de que el
    # sweep termina, para no cortar el decaimiento de la sala.
    n_cola = int(cola_s * fs)
    if n_cola > 0:
        playback = np.concatenate([sweep.astype(np.float64), np.zeros(n_cola)])
    else:
        playback = sweep.astype(np.float64)

    # sounddevice espera shape (n, channels) para playrec.
    playback_2d = playback.reshape(-1, 1)

    print("Playing sweep and recording microphone simultaneously...")
    print("  (Keep the volume moderate to avoid clipping / feedback)")

    try:
        grabacion = sd.playrec(
            playback_2d,
            samplerate=fs,
            channels=1,
            dtype="float64",
            input_device=dispositivo_entrada,
            output_device=dispositivo_salida,
        )
        sd.wait()  # Bloquea hasta que termine el stream dúplex.
    except sd.PortAudioError as exc:
        raise RuntimeError(
            "Falló la reproducción/grabación (PortAudio). "
            "Causas típicas: mic o parlante en uso por otra app, "
            "fs no soportada por el dispositivo, o permisos de micrófono "
            "denegados en Windows. "
            f"Detalle: {exc}"
        ) from exc

    # playrec devuelve (n_muestras, canales) → aplanamos a mono 1-D.
    return np.asarray(grabacion[:, 0], dtype=np.float64)


def guardar_grabacion_wav(senal: np.ndarray, fs: int, ruta: Path) -> None:
    """
    Guarda la grabación como WAV PCM 16-bit.

    QUÉ: Convierte float [-1, 1] a int16 y escribe el archivo.
    POR QUÉ: Mismo formato que el sweep; facilita comparar e importar
    en fases siguientes sin conversiones raras.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    senal_clip = np.clip(senal, -1.0, 1.0)
    pcm = (senal_clip * 32767.0).astype(np.int16)
    wavfile.write(str(ruta), fs, pcm)


def graficar_grabacion(senal: np.ndarray, fs: int, ruta: Path) -> None:
    """
    Guarda un gráfico simple de la forma de onda grabada.

    QUÉ: Amplitud vs tiempo → PNG.
    POR QUÉ: Verificación rápida: ¿se ve el sweep? ¿hay clipping
    (picos pegados a ±1)? ¿el mic capturó algo o quedó en silencio?
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(senal)) / fs

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, senal, linewidth=0.4, color="#1a1a1a")
    ax.set_title("Recorded response (mic)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(t[0], t[-1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
