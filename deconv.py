"""
Deconvoluciona la grabación con el filtro inverso del sweep (Farina).
Obtiene la respuesta al impulso (IR) → resultados/impulso.wav
"""

from pathlib import Path

import numpy as np
from scipy.io import wavfile

# Mismos parámetros con los que se generó el sweep
fInicio = 45
fFin = 20000

fs, grabacion = wavfile.read("resultados/sweep+sala.wav")
fs, sweep = wavfile.read("resultados/sweep.wav")

# Paso a float para calcular
sweep = sweep.astype(np.float32)
grabacion = grabacion.astype(np.float32)

n = len(sweep)
T = n / fs
t = np.linspace(0, T, n)
L = np.log(fFin / fInicio)

# Envolvente: compensa que el sweep log deja más energía en graves
envolvente = np.exp(-t * L / T)

# Filtro inverso = sweep al revés × envolvente
filtro = sweep[::-1] * envolvente

# Convolución: aplico el filtro a la grabación → IR (la "palmada" del sistema)
impulso = np.convolve(grabacion, filtro)

print(f"Pico IR: {np.max(np.abs(impulso)):.4f}")
print(f"Largo IR: {len(impulso)}")

carpeta = Path("resultados")
carpeta.mkdir(parents=True, exist_ok=True)
wavfile.write(str(carpeta / "impulso.wav"), fs, impulso.astype(np.float32))
print(f"Impulso guardado en {carpeta / 'impulso.wav'}")
