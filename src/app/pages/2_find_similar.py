"""
Página 2: Find Similar
Encuentra las 5 canciones más parecidas en el dataset real
dadas las características de audio del usuario.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Find Similar | Sound DNA",
    page_icon="🔍",
    layout="wide"
)

# ─── FEATURES PARA SIMILITUD ─────────────────────────────────────────────────
# Para buscar canciones parecidas usamos solo las features de AUDIO (no Unnamed:0,
# que es un proxy de género/índice y no representa audio).

AUDIO_FEATURES = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'time_signature', 'duration_min'
]

MODEL_FEATURES = [
    'Unnamed: 0', 'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'time_signature', 'duration_min'
]

FEATURE_INFO = {
    'danceability':     ('💃 Bailabilidad',      0.0,  1.0,  0.60, 0.01),
    'energy':           ('⚡ Energía',            0.0,  1.0,  0.65, 0.01),
    'key':              ('🎹 Tonalidad',          0,    11,   5,    1),
    'loudness':         ('🔊 Volumen (dB)',       -60,  0,    -7.0, 0.1),
    'mode':             ('🎵 Modo',               0,    1,    1,    1),
    'speechiness':      ('🎤 Voz hablada',        0.0,  1.0,  0.05, 0.01),
    'acousticness':     ('🎸 Acústica',           0.0,  1.0,  0.20, 0.01),
    'instrumentalness': ('🎹 Instrumental',       0.0,  1.0,  0.05, 0.01),
    'liveness':         ('🎭 En directo',         0.0,  1.0,  0.12, 0.01),
    'valence':          ('😊 Valencia',           0.0,  1.0,  0.50, 0.01),
    'tempo':            ('🥁 Tempo (BPM)',        50,   220,  120,  1),
    'time_signature':   ('🎼 Compás',             3,    5,    4,    1),
    'duration_min':     ('⏱️ Duración (min)',     0.5,  8.0,  3.5,  0.1),
}

# ─── CARGA DE DATOS Y MODELO KNN ─────────────────────────────────────────────

@st.cache_data
def load_dataset():
    """
    Carga train.csv y test.csv y los une para buscar en todo el dataset.
    El decorador @st.cache_data hace que solo se cargue una vez (no en cada interacción).
    """
    data_path = Path(__file__).resolve().parents[3] / "data" / "processed" / "spotify_clean.csv"
    

    dfs = []
    for p in [data_path]:
        if p.exists():
            dfs.append(pd.read_csv(p))

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)

    # Nos quedamos solo con las filas que tienen las columnas necesarias
    required = AUDIO_FEATURES + ['popularity']
    df = df.dropna(subset=required)

    return df


@st.cache_resource
def build_knn_index(df_hash):
    """
    Construye el índice KNN sobre las features de audio.
    KNN (K-Nearest Neighbors) es un algoritmo que, dado un punto nuevo,
    encuentra los K puntos más cercanos en un espacio de características.
    Como un "vecino más próximo": si quiero saber qué canciones suenan como la mía,
    busco las que tienen características de audio más parecidas.
    """
    df = load_dataset()
    if df is None:
        return None, None

    X = df[AUDIO_FEATURES].values

    # StandardScaler: normaliza cada feature para que tengan la misma escala.
    # Sin esto, el volumen en dB (-60 a 0) dominaría sobre bailabilidad (0 a 1).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KNN con 10 vecinos y distancia euclidiana
    knn = NearestNeighbors(n_neighbors=10, metric='euclidean')
    knn.fit(X_scaled)

    return knn, scaler


# ─── UI PRINCIPAL ─────────────────────────────────────────────────────────────

st.markdown("# 🔍 Find Similar")
st.markdown("""
Define las características de audio de tu canción y el algoritmo encontrará
las 5 más parecidas entre las 114,000 del dataset.
""")

# Modo de búsqueda
search_mode = st.radio(
    "Modo de búsqueda:",
    ["🎚️ Por características (sliders)", "🔤 Por nombre de canción"],
    horizontal=True
)

st.markdown("---")

df = load_dataset()

if df is None:
    st.error("⚠️ No se encontraron src/data/train.csv o src/data/test.csv")
    st.stop()

feature_values = {}

if search_mode == "🎚️ Por características (sliders)":

    col1, col2, col3 = st.columns(3)
    cols_cycle = [col1, col2, col3]

    for i, (feat, (label, min_v, max_v, default, step)) in enumerate(FEATURE_INFO.items()):
        with cols_cycle[i % 3]:
            if isinstance(step, int):
                val = st.slider(label, int(min_v), int(max_v), int(default))
            else:
                val = st.slider(label, float(min_v), float(max_v), float(default), step=float(step))
            feature_values[feat] = val

else:  # Búsqueda por nombre
    # Detectar si existen columnas de nombre de canción
    name_col  = 'track_name'  if 'track_name'  in df.columns else None
    artist_col = 'artists'    if 'artists'      in df.columns else None

    if name_col is None:
        st.warning("El dataset no tiene columna 'track_name'. Usa el modo de sliders.")
        st.stop()

    # Crear lista de canciones para el selectbox
    if artist_col:
        df['display'] = df[artist_col].astype(str) + " — " + df[name_col].astype(str)
    else:
        df['display'] = df[name_col].astype(str)

    song_search = st.text_input("Busca una canción (escribe y selecciona):", "Blinding Lights")
    matches = df[df['display'].str.contains(song_search, case=False, na=False)].head(20)

    if matches.empty:
        st.warning("No se encontró esa canción en el dataset. Prueba otro nombre.")
        st.stop()

    selected = st.selectbox("Selecciona una canción:", matches['display'].tolist())
    row = df[df['display'] == selected].iloc[0]

    for feat in AUDIO_FEATURES:
        feature_values[feat] = float(row[feat])

    st.markdown("**Características de la canción seleccionada:**")
    preview_cols = st.columns(4)
    feats_preview = ['danceability', 'energy', 'valence', 'tempo']
    for j, f in enumerate(feats_preview):
        preview_cols[j].metric(FEATURE_INFO[f][0], f"{feature_values[f]:.2f}")

st.markdown("---")

# ─── BUSCAR SIMILARES ─────────────────────────────────────────────────────────

if st.button("🔍  Buscar canciones similares", use_container_width=True, type="primary"):

    # Hash del dataframe para usar como clave de caché del KNN
    df_hash = len(df)
    knn, scaler = build_knn_index(df_hash)

    if knn is None:
        st.error("No se pudo construir el índice de búsqueda.")
        st.stop()

    # Preparar vector de la canción buscada
    query_vec = np.array([[feature_values[f] for f in AUDIO_FEATURES]])
    query_scaled = scaler.transform(query_vec)

    # Buscar los 10 vecinos más cercanos (pedimos 10 para poder filtrar duplicados)
    distances, indices = knn.kneighbors(query_scaled)

    results = df.iloc[indices[0]].copy()
    results['similitud'] = 1 / (1 + distances[0])  # convertir distancia en score 0-1

    # Mostrar top 5
    st.markdown("## 🎵 Las 5 canciones más similares")

    display_cols = []
    if 'artists'    in results.columns: display_cols.append('artists')
    if 'track_name' in results.columns: display_cols.append('track_name')
    if 'track_genre' in results.columns: display_cols.append('track_genre')
    display_cols += ['popularity', 'danceability', 'energy', 'valence', 'tempo', 'similitud']
    display_cols = [c for c in display_cols if c in results.columns]

    top5 = results.head(5)[display_cols].reset_index(drop=True)
    top5['similitud'] = top5['similitud'].map(lambda x: f"{x*100:.1f}%")
    top5.index = range(1, 6)

    st.dataframe(top5, use_container_width=True)

    st.caption("""
    **¿Cómo funciona la similitud?** El algoritmo KNN (K-Nearest Neighbors) normaliza
    todas las features a la misma escala y calcula la distancia euclidiana entre canciones.
    Una canción con 95% similitud tiene características de audio casi idénticas a la buscada.
    """)
