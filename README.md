# Respuesta en frecuencia acústica

Proyecto en Python para medir **cómo responde un sistema de audio** (parlantes + sala + micrófono): qué frecuencias refuerza o atenúa.

No es un medidor de laboratorio calibrado. Es una herramienta de aprendizaje / portfolio, pensada también como diagnóstico básico para un home studio.

---

## Idea en corto

1. Genero un **sine sweep** (un tono que barre de graves a agudos).
2. Lo reproduzco por los parlantes y lo grabo con el mic **al mismo tiempo**.
3. Con un filtro inverso (método de Farina) y una convolución obtengo la **respuesta al impulso (IR)** — la “palmada” del sistema con su cola de eco.
4. A esa IR le hago una **FFT** y grafico la **respuesta en frecuencia** (dB vs Hz).

¿Por qué un sweep y no un click real? El sweep mete mucha energía, frecuencia por frecuencia, y se mide mejor. Después calculamos cómo habría sido la respuesta a un click ideal.

---

## Requisitos

- Python 3.10+
- Parlantes (o auriculares) y un micrófono
- Dependencias:

```bash
pip install -r requirements.txt
```

(`numpy`, `scipy`, `matplotlib`, `sounddevice`)

---

## Cómo correrlo

Los scripts se ejecutan **en orden**, desde la raíz del proyecto:

```bash
python generar_sweep.py   # 1) crea resultados/sweep.wav
python captura.py         # 2) reproduce + graba → resultados/sweep+sala.wav
python deconv.py          # 3) IR → resultados/impulso.wav
python analisis_fr.py     # 4) FFT → resultados/respuesta_frecuencia.png
```

### Dispositivos de audio (`captura.py`)

Antes de grabar, listá los dispositivos:

```python
import sounddevice as sd
print(sd.query_devices())
```

En este proyecto los índices por defecto son:

- entrada (mic): `1`
- salida (parlantes): `5`

Cambialos en `captura.py` según tu PC. Volumen moderado para no saturar.

---

## Qué hay en `resultados/`

| Archivo | Qué es |
|---------|--------|
| `sweep.wav` | El estímulo que se generó |
| `sweep+sala.wav` | Lo que grabó el mic (sweep + sala + sistema) |
| `impulso.wav` | Respuesta al impulso |
| `respuesta_frecuencia.png` | Gráfico dB vs frecuencia |

---

## Estructura

```text
.
├── generar_sweep.py   # ESS logarítmico + fade
├── captura.py         # play + record (sounddevice)
├── deconv.py          # filtro inverso + convolución → IR
├── analisis_fr.py     # FFT → respuesta en frecuencia
├── requirements.txt
└── resultados/
```

---

## Límites (honestos)

- Medís la **cadena completa**: parlantes → sala → mic, no “la sala sola”.
- Los niveles son **relativos** (0 dB = pico de esa medición), no dB SPL calibrados.
- Un mic de notebook colorea y mete ruido: la curva sirve como orientación, no como medición profesional.
- Mover el mic o los parlantes cambia el resultado.

Útil para ver si el monitoreo de un home studio está muy coloreado o para practicar el pipeline ESS → IR → FFT.
