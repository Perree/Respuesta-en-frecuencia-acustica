# Respuesta en frecuencia acústica — ESS / Farina

Proyecto de portfolio en Python que mide la **respuesta en frecuencia** de un sistema acústico (parlantes + sala + micrófono) usando un **sine sweep logarítmico** (Exponential Sine Sweep, ESS / método de Farina) y **deconvolución**.

Corre en una **PC con Windows** con la placa de sonido integrada (o USB): salida por parlantes/auriculares y entrada por micrófono, vía `sounddevice`.

---

## Qué mide (y por qué)

Cuando reproducís un estímulo conocido por los parlantes y lo grabás con el mic, la grabación **no** es “la sala sola”: es toda la cadena:

**estímulo → DAC / parlantes → aire / sala → mic → ADC**

Este proyecto recupera dos vistas útiles de esa cadena:

1. **Respuesta al impulso (IR)** — cómo reacciona el sistema a un “click” ideal. Se obtiene deconvolucionando el sweep grabado con el filtro inverso de Farina.
2. **Respuesta en frecuencia** — magnitud vs frecuencia (FFT de la IR), en dB. Muestra qué bandas se refuerzan o se atenúan.
3. **RT60 (opcional)** — estimación del tiempo de reverberación a partir de la curva de decaimiento energético (integración de Schroeder), cuando el SNR lo permite.

**¿Por qué ESS / Farina?** Un sweep logarítmico dedica el mismo tiempo relativo a cada octava (más cercano a cómo oímos), deposita energía en todo el rango y — tras el filtrado inverso — separa en el tiempo el impulso lineal de los productos de distorsión armónica. Por eso es un estímulo estándar y robusto para medir la función de transferencia acústica con hardware de PC cotidiano.

Es una herramienta de **aprendizaje / portfolio**, no un analizador de laboratorio calibrado.

---

## Requisitos

### Hardware (PC)

- Una computadora con placa de sonido funcionando (Realtek integrada / similar alcanza)
- Parlantes o auriculares como **salida por defecto** (o elegí un índice de dispositivo en `main.py`)
- Un micrófono como **entrada por defecto** (mic de notebook, headset o mic USB)
- Ambiente razonablemente silencioso (menos ruido de fondo → IR / RT60 más limpios)

### Software

- Python 3.10+ recomendado
- Paquetes listados en `requirements.txt`:

```text
numpy
scipy
matplotlib
sounddevice
```

`sounddevice` habla con el stack de audio del SO a través de PortAudio (en Windows, las wheels suelen traer lo necesario).

---

## Instalación

Desde la raíz del proyecto:

```bash
python -m venv .venv
```

Activá el venv (PowerShell en Windows):

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalá las dependencias:

```bash
pip install -r requirements.txt
```

---

## Cómo correrlo (en tu PC)

1. Poné el volumen de Windows en un nivel cómodo (evitá clipping / distorsión).
2. Asegurate de que el mic que querés usar sea el dispositivo de grabación por defecto (o anotá su índice).
3. Desde la raíz del proyecto:

```bash
python main.py
```

Qué hace el pipeline:

1. Genera un sine sweep logarítmico → `resultados/sweep.wav` (+ gráfico de waveform)
2. Lista los dispositivos de audio y luego **reproduce** el sweep y **graba** el mic → `resultados/grabacion.wav`
3. Construye el filtro inverso de Farina y deconvoluciona → respuesta al impulso
4. FFT → gráfico de respuesta en frecuencia; opcionalmente estima RT60

### Variables útiles en `main.py`

| Variable | Significado |
|----------|-------------|
| `usar_grabacion_existente` | `True` reutiliza `resultados/grabacion.wav` (sin regrabar) para iterar el análisis |
| `dispositivo_entrada` / `dispositivo_salida` | `None` = default de Windows/PortAudio; poné un índice según la lista impresa |
| `estimar_rt60` | `False` saltea RT60 y solo calcula la respuesta en frecuencia |
| `metodo_rt60` | `"T30"` o `"T20"` (usá T20 si el SNR no alcanza para T30) |

Mirás en consola el **recording peak**: valores cerca de `1.0` indican clipping — bajá el volumen de Windows o `amplitud` y volvé a grabar.

---

## Qué hay en `resultados/`

Después de una corrida completa deberías ver algo así:

| Archivo | Qué representa |
|---------|----------------|
| `sweep.wav` | El estímulo ESS que se generó y se reprodujo |
| `sweep_waveform.png` | Gráfico temporal del sweep (chequeo rápido) |
| `grabacion.wav` | Grabación del mic mientras sonaba el sweep (respuesta del sistema + ruido) |
| `grabacion_waveform.png` | Gráfico temporal de esa grabación (nivel / clipping) |
| `impulso.wav` | Respuesta al impulso tras la deconvolución (PCM para escuchar / otras herramientas) |
| `impulso.npy` | La misma IR en float64 (precisión completa para seguir analizando) |
| `impulso.png` | Gráfico temporal de la IR — pico de sonido directo + decaimiento / reflexiones |
| `respuesta_frecuencia.png` | Espectro de magnitud de la IR en dB vs frecuencia log (el gráfico principal de respuesta en frecuencia) |
| `decaimiento_rt60.png` | Curva de decaimiento energético + ajuste de RT60 (solo si la estimación de RT60 funciona) |

Si se saltea el RT60, la consola explica el motivo (a menudo SNR insuficiente con mic de laptop).

---

## Mapa de módulos (`src/`)

| Módulo | Rol |
|--------|-----|
| `generar_sweep.py` | Genera el ESS logarítmico (fórmula de Farina), guarda WAV y grafica el waveform |
| `grabar_audio.py` | Lista dispositivos; reproduce + graba en simultáneo con `sounddevice`; guarda la captura |
| `deconvolucion.py` | Filtro inverso + convolución FFT → respuesta al impulso; helpers de carga/guardado |
| `analisis.py` | Respuesta en frecuencia por FFT; RT60 opcional (EDC de Schroeder) y gráfico de decaimiento |

`main.py` en la raíz del proyecto orquesta estas cuatro etapas de punta a punta y escribe todo bajo `resultados/`.

---

## Limitaciones honestas

- **Sin calibración absoluta.** Los niveles son relativos (0 dB = pico espectral de esa medición). No podés afirmar precisión en dB SPL ni “flatness” absoluta sin mic calibrado y referencia.
- **Los mics integrados de laptop son ruidosos y coloreados.** Atenúan graves, suman hiss y suelen estar cerca de coolers/teclado. Ese sesgo aparece en la respuesta en frecuencia y en la IR.
- **Loop acústico, no loopback eléctrico.** Medís parlantes + sala + mic juntos. Las reflexiones, los muebles y la posición del mic dominan el resultado: movés el mic y la curva cambia.
- **Rarezas de reloj / dispositivos.** Distintas frecuencias de muestreo, modo exclusivo o Bluetooth pueden distorsionar el timing o el ancho de banda. Preferí parlantes/auriculares cableados y un `fs` estable (por defecto 48 kHz).
- **El RT60 es frágil acá.** El método de Schroeder necesita un decaimiento limpio y rango dinámico suficiente. Con SNR de laptop a menudo solo alcanzan estimaciones T20 burdas: tomá los números como educativos, no como acústica de sala tipo ISO.
- **Distorsión y no linealidad.** Farina ayuda a separar armónicos en el tiempo, pero el clipping fuerte, el AGC o el DSP “enhance” del SO/driver igual corrompen la IR lineal.

Usá este proyecto para **demostrar el pipeline ESS y el razonamiento DSP**, no como sustituto de equipo profesional de medición (mics calibrados, interfaces, REW, etc.).

---

## Estructura del proyecto

```text
.
├── main.py              # Orquesta las fases 1–4
├── requirements.txt
├── README.md
├── src/
│   ├── generar_sweep.py
│   ├── grabar_audio.py
│   ├── deconvolucion.py
│   └── analisis.py
└── resultados/          # WAVs, .npy y gráficos (se generan al correr)
```
