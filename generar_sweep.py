"""
Genera un sine sweep logarítmico (ESS) y lo guarda en resultados/sweep.wav.
"""

from pathlib import Path

import numpy as np
from scipy.io import wavfile

# Parámetros de la señal de audio
fInicio = 45  # Hz
fFin = 20000  # Hz
s = 10  # segundos
fs = 48000  # Hz
amp = 0.5  # amplitud

# Reloj del sistema: una marca de tiempo por muestra
n_muestras = int(fs * s)
t = np.linspace(0, s, n_muestras)

# Frecuencia en cada instante (mapa del barrido 45 Hz → 20 kHz)
f_t = fInicio * (fFin / fInicio) ** (t / s)

# Escala log del barrido (el oído trabaja por octavas / ratios)
L = np.log(fFin / fInicio)

# Fase de Farina → el seno cuya frecuencia sube con el tiempo
fase = (2 * np.pi * fInicio * s) / L * (np.exp(t * L / s) - 1)
sweep = amp * np.sin(fase)

# Fade in / fade out (anti-click)
n_fade = int(0.020 * fs)
fade = np.linspace(0, 1, n_fade)  # rampa de 0 a 1 en 20 ms
sweep[:n_fade] *= fade
sweep[-n_fade:] *= fade[::-1]

# Guardo el estímulo
carpeta = Path("resultados")
carpeta.mkdir(parents=True, exist_ok=True)
wavfile.write(str(carpeta / "sweep.wav"), fs, sweep.astype(np.float32))
print(f"Sweep guardado en {carpeta / 'sweep.wav'}")
