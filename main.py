"""
Punto de entrada — Fase 1 + Fase 2 + Fase 3 + Fase 4.

1) Genera el sine sweep logarítmico (ESS).
2) Lo reproduce+graba en la PC, o reutiliza una grabación ya guardada.
3) Deconvoluciona con el filtro inverso de Farina → respuesta al impulso.
4) Analiza la IR: FFT (respuesta en frecuencia) y RT60 opcional.
"""

from pathlib import Path
import sys

import numpy as np

# Permite `from generar_sweep import ...` sin instalar el paquete.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from generar_sweep import (
    generar_sweep_logaritmico,
    guardar_sweep_wav,
    graficar_waveform,
)
from grabar_audio import (
    listar_dispositivos,
    reproducir_y_grabar,
    guardar_grabacion_wav,
    graficar_grabacion,
)
from deconvolucion import (
    construir_filtro_inverso,
    deconvolucionar,
    cargar_wav_mono,
    guardar_impulso,
    graficar_impulso,
)
from analisis import (
    calcular_respuesta_frecuencia,
    graficar_respuesta_frecuencia,
    calcular_rt60,
    graficar_decaimiento,
)


def main() -> None:
    """Orquesta estímulo → grabación → IR → análisis espectral / RT60."""
    # --- Parámetros del estímulo (Fase 1) ---
    f_inicio = 20.0
    f_fin = 20000.0
    duracion = 8.0
    fs = 48000
    amplitud = 0.5

    # --- Dispositivos de audio (Fase 2) ---
    # Índices MME según `sounddevice.query_devices()` en esta máquina.
    dispositivo_entrada = 1  # Línea de entrada (BEHRINGER)
    dispositivo_salida = 6   # VoiceMeeter Aux Input → ruteá a parlantes en VoiceMeeter
    cola_s = 1.5             # silencio extra al final para capturar la sala

    # True  → carga resultados/grabacion.wav (iterar sin regrabar).
    # False → reproduce el sweep y graba el mic de nuevo.
    usar_grabacion_existente = False

    # --- Análisis (Fase 4) ---
    estimar_rt60 = True   # False → solo FFT / respuesta en frecuencia
    metodo_rt60 = "T30"   # "T20" si el SNR no alcanza para T30

    carpeta = ROOT / "resultados"
    ruta_sweep = carpeta / "sweep.wav"
    ruta_sweep_png = carpeta / "sweep_waveform.png"
    ruta_grabacion = carpeta / "grabacion.wav"
    ruta_grabacion_png = carpeta / "grabacion_waveform.png"
    ruta_impulso_wav = carpeta / "impulso.wav"
    ruta_impulso_npy = carpeta / "impulso.npy"
    ruta_impulso_png = carpeta / "impulso.png"
    ruta_fr_png = carpeta / "respuesta_frecuencia.png"
    ruta_rt60_png = carpeta / "decaimiento_rt60.png"

    # --- Fase 1: generar estímulo ---
    print("Generating logarithmic sine sweep (ESS / Farina)...")
    sweep = generar_sweep_logaritmico(
        f_inicio=f_inicio,
        f_fin=f_fin,
        duracion=duracion,
        fs=fs,
        amplitud=amplitud,
    )
    guardar_sweep_wav(sweep, fs, ruta_sweep)
    graficar_waveform(sweep, fs, ruta_sweep_png)
    print(f"  WAV  : {ruta_sweep}")
    print(f"  Plot : {ruta_sweep_png}")
    print()

    # --- Fase 2: grabación (nueva o existente) ---
    if usar_grabacion_existente:
        if not ruta_grabacion.exists():
            raise FileNotFoundError(
                f"No está {ruta_grabacion}. "
                "Poné usar_grabacion_existente=False para grabar primero."
            )
        print(f"Loading existing recording: {ruta_grabacion}")
        grabacion, fs_grab = cargar_wav_mono(ruta_grabacion)
        if fs_grab != fs:
            raise ValueError(
                f"La grabación tiene fs={fs_grab} pero el sweep usa fs={fs}. "
                "Regenerá/regrabá con la misma frecuencia de muestreo."
            )
    else:
        listar_dispositivos()
        grabacion = reproducir_y_grabar(
            sweep,
            fs,
            dispositivo_entrada=dispositivo_entrada,
            dispositivo_salida=dispositivo_salida,
            cola_s=cola_s,
        )
        guardar_grabacion_wav(grabacion, fs, ruta_grabacion)
        graficar_grabacion(grabacion, fs, ruta_grabacion_png)

        pico = float(np.max(np.abs(grabacion))) if len(grabacion) else 0.0
        print(f"  Recording peak : {pico:.4f} (1.0 = full scale / clipping)")
        print(f"  WAV            : {ruta_grabacion}")
        print(f"  Plot           : {ruta_grabacion_png}")
        print()

    # --- Fase 3: filtro inverso + deconvolución → IR ---
    print("Building inverse filter and deconvolving...")
    filtro = construir_filtro_inverso(sweep, f_inicio, f_fin, fs)
    impulso = deconvolucionar(grabacion, filtro)
    guardar_impulso(impulso, fs, ruta_impulso_wav, ruta_impulso_npy)
    graficar_impulso(impulso, fs, ruta_impulso_png)

    pico_ir = float(np.max(np.abs(impulso)))
    print(f"  IR peak        : {pico_ir:.6f}")
    print(f"  WAV            : {ruta_impulso_wav}")
    print(f"  NPY (float64)  : {ruta_impulso_npy}")
    print(f"  Plot           : {ruta_impulso_png}")
    print()

    # --- Fase 4: FFT + RT60 opcional ---
    print("Analyzing impulse response (FFT)...")
    frecuencias, magnitud_db = calcular_respuesta_frecuencia(
        impulso,
        fs,
        f_min=f_inicio,
        f_max=min(f_fin, fs / 2.0),
    )
    graficar_respuesta_frecuencia(frecuencias, magnitud_db, ruta_fr_png)
    print(f"  Frequency response plot : {ruta_fr_png}")

    if estimar_rt60:
        print(f"Estimating RT60 ({metodo_rt60})...")
        try:
            rt60_s, tiempos, edc_db, info = calcular_rt60(
                impulso,
                fs,
                metodo=metodo_rt60,
            )
            graficar_decaimiento(tiempos, edc_db, info, rt60_s, ruta_rt60_png)
            print(f"  RT60 ({metodo_rt60}) : {rt60_s:.3f} s")
            print(f"  Decay plot         : {ruta_rt60_png}")
        except RuntimeError as exc:
            # Con mic/parlantes de laptop el SNR a veces no alcanza T30.
            print(f"  RT60 skipped: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
