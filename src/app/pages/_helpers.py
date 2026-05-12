"""
Helpers compartidos por las páginas Streamlit que usan los modelos
==================================================================
Centraliza tres cosas que de otra forma estarían duplicadas en cada página:

1. La lista exacta de columnas que esperan los modelos (orden incluido).
2. La constante MAE del regresor, que se usa para mostrar el margen de error.
3. La función load_models, decorada con @st.cache_resource para que solo se
   cargue una vez por sesión.

Si en el futuro re-entrenas los modelos con otras features, solo tocas
MODEL_FEATURES aquí y todas las páginas se actualizan.

Streamlit ignora los archivos cuyo nombre empieza por '_' como páginas, así
que este archivo NO aparece en el menú lateral aunque viva dentro de pages/.
"""

from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Rutas
# -----------------------------------------------------------------------------
# Este archivo vive en sound-dna/src/app/pages/_helpers.py
# parents[0] = pages/ ; parents[1] = app/ ; parents[2] = src/
ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "model" / "production"


# -----------------------------------------------------------------------------
# Constantes del modelo
# -----------------------------------------------------------------------------
# MAE del regresor en el set de test (notebook 04, 5 may 2026).
# Se usa para mostrar al usuario un rango de error: predicción ± MAE puntos.
REGRESSION_MAE = 7.85

# Orden EXACTO de features que los modelos vieron durante el entrenamiento.
# Confirmado con: m.feature_names_in_ sobre los .pkl del 5 de mayo.
# IMPORTANTE: si cambias este orden o las columnas, los modelos romperán.
# "Unnamed: 0" es la columna índice residual del CSV original; el modelo
# la espera pero su valor real no afecta a la predicción (siempre la
# mandamos a 0).
MODEL_FEATURES = [
    "Unnamed: 0",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
    "duration_min",
]


# -----------------------------------------------------------------------------
# Carga de modelos
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Carga los dos modelos serializados desde disco.

    Devuelve la tupla (clf, reg). Si falta alguno de los archivos, devuelve
    (None, None) y la página debe mostrar un error.

    El decorador @st.cache_resource garantiza que esta función solo se
    ejecuta una vez por sesión (por usuario): la primera carga puede tardar
    unos segundos, las siguientes son instantáneas.
    """
    try:
        clf = joblib.load(MODEL_DIR / "modelo_clasificacion_final.pkl")
        reg = joblib.load(MODEL_DIR / "modelo_regresion_final.pkl")
        return clf, reg
    except FileNotFoundError:
        return None, None


# -----------------------------------------------------------------------------
# Construcción del input para los modelos
# -----------------------------------------------------------------------------
def build_model_input(features: dict) -> pd.DataFrame:
    """Construye el DataFrame que los modelos esperan.

    - Devuelve un DataFrame de 1 fila con las 14 columnas en el orden
      correcto (el de MODEL_FEATURES).
    - Si el dict de entrada no incluye `Unnamed: 0`, se añade con valor 0.
    - Si falta `time_signature`, se rellena con 4 (compás 4/4, el más común
      en pop).
    - Cualquier columna extra del dict (por ejemplo `explicit`) se ignora:
      el modelo no la conoce y rompería si se la pasáramos.
    """
    sensible_defaults = {
        "Unnamed: 0": 0,
        "time_signature": 4,
    }

    row = {
        col: features.get(col, sensible_defaults.get(col, 0))
        for col in MODEL_FEATURES
    }
    row["Unnamed: 0"] = 0  # forzamos a 0 siempre, sea lo que sea que llegue

    return pd.DataFrame([row], columns=MODEL_FEATURES)
