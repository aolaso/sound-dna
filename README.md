# Sound DNA — La anatomía de audio de un hit

> *¿Se puede predecir si una canción será un éxito solo con su sonido?*

Proyecto final de Machine Learning · The Bridge Data Science Bootcamp 2026  
Andoni · Mayo 2026

---

## El proyecto

Sound DNA entrena dos modelos sobre 114.000 canciones de Spotify para intentar responder una pregunta concreta: **¿cuánto explica el audio de una canción su popularidad?**

El resultado: el sonido explica el **61% de la varianza de popularidad**. El 40% restante — marketing, nombre del artista, momento cultural, el algoritmo de Spotify — vive fuera del audio. Eso no es un fracaso del modelo. Es el hallazgo principal.

**Dos tareas en paralelo:**
- **Regresión** — predice la puntuación exacta de popularidad (0–100). Resultado: MAE 7.85, R² 0.613.
- **Clasificación** — predice si la canción será un *hit* (popularidad ≥ 70, el 4.8% del dataset). Resultado: F1 0.211, AUC 0.801, Recall 0.467.

---

## Resultados del modelo final

| Tarea | Métrica | Valor |
|---|---|---|
| Regresión | MAE | **7.85** |
| Regresión | R² | **0.613** |
| Clasificación | F1 | **0.211** |
| Clasificación | AUC | **0.801** |
| Clasificación | Recall | **0.467** |

Modelo: `RandomForestRegressor` / `RandomForestClassifier` · Pipeline con `StandardScaler`  
Hiperparámetros: `n_estimators=50`, `max_depth=15`, `min_samples_split=2`

La sorpresa: las features más predictivas **no son las intuitivas**. El modelo dice que los hits tienen voz humana (instrumentalness baja), duración corta, y un balance óptimo entre lo cantado y lo hablado. Energía y bailabilidad quedan al fondo.

---

## Estructura del repositorio

```
sound-dna/
├── .gitignore
├── README.md
├── requirements.txt
│
└── src/
    ├── memoria.ipynb              ← Resumen ejecutivo del proyecto (I–VI)
    │
    ├── data/
    │   ├── raw/
    │   │   └── spotify_tracks.csv     ← Dataset original (no en git, ver abajo)
    │   ├── processed/                 ← CSVs intermedios (no en git)
    │   ├── train.csv                  ← Split de entrenamiento (80%, 60.365 filas)
    │   └── test.csv                   ← Split de test (20%, 15.092 filas)
    │
    ├── notebooks/
    │   ├── 01_eda.ipynb               ← Análisis exploratorio
    │   ├── 02_limpieza.ipynb          ← Limpieza y preparación
    │   ├── 03_baseline.ipynb          ← Modelos de referencia
    │   ├── 04_avanzados.ipynb         ← Decision Tree, Random Forest, Gradient Boosting
    │   └── 05_grid.ipynb              ← GridSearchCV, tuning, modelos finales
    │
    ├── utils/
    │   ├── audio_features.py          ← Extracción de features con librosa
    │   ├── preprocessing.py           ← Funciones de limpieza reutilizables
    │   └── viz.py                     ← Helpers de visualización
    │
    ├── model/
    │   └── production/
    │       ├── modelo_clasificacion_final.pkl   ← Clasificador (1.95 MB) ✓ en git
    │       └── modelo_regresion_final.pkl       ← Regresor (8.93 MB) — ver nota abajo
    │
    └── app/
        ├── streamlit_app.py           ← App técnica (entrega bootcamp)
        └── pages/
            ├── _helpers.py
            ├── 1_test_your_idea.py
            ├── 2_find_similar.py
            ├── 3_what_if.py
            └── 4_audio_upload_beta.py

resources/
└── img/                               ← Visualizaciones PNG (real vs predicho, ROC, etc.)
```

> **Modelo de regresión:** El archivo `modelo_regresion_final.pkl` (8.93 MB) no está incluido
> en este repositorio por tamaño. Puedes descargarlo aquí: *[añadir link Google Drive]*
> o reproducirlo ejecutando los notebooks en orden (01 → 05).

---

## Dataset

- **Fuente:** Kaggle — [Spotify Tracks Dataset](https://www.kaggle.com/)
- **Tamaño original:** 114.000 canciones · 114 géneros · 21 columnas
- **Tras limpieza:** ~67.000 canciones únicas con popularidad ≥ 10
- **Split:** 80/20 estratificado por `is_hit` (stratify garantiza 4.8% hits en ambas partes)

El dataset original (`spotify_tracks.csv`) no está en el repo por tamaño.  
`train.csv` y `test.csv` sí están incluidos (requisito de entrega).

---

## Instalación y ejecución

### Requisitos

```bash
# Clonar el repositorio
git clone https://github.com/<usuario>/sound-dna.git
cd sound-dna

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Instalar dependencias — IMPORTANTE: versiones pinadas
pip install -r requirements.txt
```

> **Aviso de versiones:** Los modelos `.pkl` se entrenaron con `scikit-learn==1.8.0`.
> Cargarlos con una versión distinta puede dar `InconsistentVersionWarning` o resultados erróneos.
> Si ocurre, restaura con: `pip install scikit-learn==1.8.0 numpy==2.3.5 pandas==2.3.3`

### Ejecutar los notebooks en orden

```bash
cd src
jupyter notebook
# Abrir y ejecutar: 01_eda → 02_limpieza → 03_baseline → 04_avanzados → 05_grid
```

Cada notebook carga los archivos que generó el anterior. El orden importa.

### Lanzar la app Streamlit

```bash
cd src/app
streamlit run streamlit_app.py
# Se abre en http://localhost:8501
```

La app carga directamente los modelos `.pkl` desde `src/model/production/`.  
Si el modelo de regresión no está en el repo, descárgalo primero (ver enlace arriba).

---

## Los cuatro modos de la app

| Modo | Descripción |
|---|---|
| **Test your idea** | Diseña una canción imaginaria con sliders y obtén su predicción de hit |
| **Find similar** | Selecciona una canción real y encuentra las más parecidas en sonido |
| **What if** | Modifica las features de una canción real y ve cómo cambia la predicción |
| **Audio upload (BETA)** | Sube un MP3/WAV y extrae sus features con librosa para predecir |

---

## Decisiones técnicas clave

**¿Por qué el umbral de hit es 70?**  
El umbral 70 da un 4.8% de hits — suficiente para que el modelo tenga ejemplos, pero suficientemente exigente para reflejar lo que un sello discográfico llamaría hit. Con umbral 50 el problema sería trivial; con 90 apenas habría ejemplos.

**¿Por qué `class_weight='balanced'`?**  
Sin balanceo, el clasificador aprende a decir "no hit" a todo y consigue 95.9% de accuracy con F1=0. Con balanceo, el modelo penaliza más los errores en hits (la clase minoritaria). La accuracy baja al 59%, el F1 sube de 0 a 0.211. Es el resultado correcto.

**¿Por qué `n_estimators=50` y no el óptimo del GridSearch?**  
El GridSearch puede sugerir 100–200 árboles como óptimo estadístico. La mejora en MAE es inferior a 1 punto sobre 100, mientras que el archivo pasa de ~9 MB a >100 MB (GitHub rechaza archivos >100 MB). n_estimators=50 es el punto de equilibrio.

**¿Qué es `Unnamed: 0`?**  
Una columna índice que se coló al guardar el CSV sin `index=False`. El modelo la usa como proxy implícito de género (las canciones están ordenadas en bloques de 1.000 por género). La solución limpia — incluir género explícitamente como one-hot — es trabajo futuro documentado en el notebook 04.

---

## Limitaciones conocidas

- El audio explica el 61% de la varianza — el 40% restante es marketing, artista y contexto
- Modelo único para 114 géneros: aprende un patrón promedio de hit (trabajo futuro: modelos por macro-género)
- "Popularidad" en Spotify mide reproducciones *recientes*, no éxito histórico
- El modo Audio upload usa `librosa`, que aproxima las features de Spotify pero no las replica exactamente

---

## Web pública

La versión consumer del proyecto está disponible en: **[sound-dna.lovable.app](https://sound-dna.lovable.app)** *(añadir URL real)*

Frontend en React (Lovable) · Backend en FastAPI · Mismo modelo `.pkl`

---

*The Bridge Data Science Bootcamp 2026 · Proyecto individual de ML*
