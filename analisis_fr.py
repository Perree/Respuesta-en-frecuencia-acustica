"""
Analiza la respuesta al impulso: FFT → respuesta en frecuencia (dB vs Hz).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

fs, impulso = wavfile.read("resultados/impulso.wav")
impulso = impulso.astype(np.float32)

# Dónde está el golpe más fuerte (sonido directo)
idx = np.argmax(np.abs(impulso))

# Recorto alrededor del pico: un poco antes + ~1 s de cola
n_pre = int(0.005 * fs)  # 5 ms
n_post = int(1.0 * fs)
i0 = max(0, idx - n_pre)
i1 = min(len(impulso), idx + n_post)
recorte = impulso[i0:i1]

# FFT del pedazo útil
espectro = np.fft.rfft(recorte)
magnitud = np.abs(espectro)
frecuencias = np.fft.rfftfreq(len(recorte), d=1.0 / fs)

# dB relativos al pico (0 dB = lo más fuerte de esta medición)
magnitud_db = 20 * np.log10(magnitud / np.max(magnitud) + 1e-12)

# Gráfico: eje X log (como se mira audio)
carpeta = Path("resultados")
carpeta.mkdir(parents=True, exist_ok=True)
ruta_png = carpeta / "respuesta_frecuencia.png"

plt.figure(figsize=(12, 5))
plt.semilogx(frecuencias, magnitud_db)
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud (dB)")
plt.title("Respuesta en frecuencia (FFT de la IR)")
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig(ruta_png, dpi=150)
plt.show()

print(f"Gráfico guardado en {ruta_png}")
