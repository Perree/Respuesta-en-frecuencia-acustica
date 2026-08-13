"""
Reproduce el sweep por los parlantes y graba el mic al mismo tiempo.
Salida: resultados/sweep+sala.wav
"""

from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io import wavfile

# Índices según sd.query_devices() en esta PC
dispositivo_entrada = 1  # mic
dispositivo_salida = 5  # parlantes

fs, sweep = wavfile.read("resultados/sweep.wav")

device = [dispositivo_entrada, dispositivo_salida]
sample_rate = fs
channels = 1

# Reproduzco el sweep y grabo a la vez
play_rec = sd.playrec(
    sweep,
    device=device,
    samplerate=sample_rate,
    channels=channels,
)

# Espero a que termine la grabación
sd.wait()

# Dejo la grabación en 1D (todas las muestras del canal del mic)
grabacion = play_rec[:, 0]

# Guardo la grabación (sweep ya pasado por parlantes + sala + mic)
carpeta = Path("resultados")
carpeta.mkdir(parents=True, exist_ok=True)
wavfile.write(str(carpeta / "sweep+sala.wav"), sample_rate, grabacion.astype(np.float32))
print(f"Grabación guardada en {carpeta / 'sweep+sala.wav'}")
print(f"Pico de grabación: {np.max(np.abs(grabacion)):.4f}")
