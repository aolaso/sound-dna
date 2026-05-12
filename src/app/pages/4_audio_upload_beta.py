"""
Página 4: Audio Upload BETA
El usuario sube un archivo de audio (MP3/WAV).
Se extraen las features que se pueden calcular con librosa.
Las features que Spotify calcula internamente (no reproducibles exactamente)
se rellenan con sliders para que el usuario las ajuste.
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Audio Upload BETA | Sound DNA",
    page_icon="🎙️",
    layout="wide"
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

MODEL_FEATURES = [
    'Unnamed: 0', 'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'time_signature', 'duration_min'
]

# Features que SÍ podemos estimar con librosa
EXTRACTABLE = ['tempo', 'loudness', 'duration_min', 'key', 'mode', 'instrumentalness']

# Features que NO se pueden extraer exactamente (son métricas propietarias de Spotify)
# El usuario las ajusta manualmente con sliders
MANUAL_FEATURES = {
    'danceability':  ('💃 Bailabilidad (ajuste manual)',  0.0, 1.0, 0.60, 0.01),
    'energy':        ('⚡ Energía (ajuste manual)',        0.0, 1.0, 0.65, 0.01),
    'speechiness':   ('🎤 Voz hablada (ajuste manual)',   0.0, 1.0, 0.05, 0.01),
    'acousticness':  ('🎸 Acústica (ajuste manual)',      0.0, 1.0, 0.20, 0.01),
    'liveness':      ('🎭 En directo (ajuste manual)',    0.0, 1.0, 0.12, 0.01),
    'valence':       ('😊 Valencia (ajuste manual)',      0.0, 1.0, 0.50, 0.01),
    'time_signature':('🎼 Compás (ajuste manual)',        3,   5,   4,    1),
}

# ─── CARGA DE MODELOS ─────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    model_dir = Path("src/model/production")
    if not model_dir.exists():
        return None, None
    files = sorted(model_dir.glob("*.joblib"), key=lambda f: f.stat().st_size, reverse=True)
    if len(files) < 2:
        return None, None
    return joblib.load(files[0]), joblib.load(files[1])

# ─── EXTRACCIÓN DE FEATURES CON LIBROSA ──────────────────────────────────────

def extract_features_librosa(audio_bytes):
    """
    Librosa es una librería de Python para análisis de audio.
    Extrae características matemáticas de la señal de sonido:
    - Tempo: cuenta los picos de energía para estimar BPM
    - Loudness: calcula el nivel de energía RMS (Root Mean Square)
    - Key/Mode: usa Chroma features (distribución de notas musicales)
    - Duration: longitud total del archivo
    - Instrumentalness: estimada como inverso de presencia de voz (MFCC)
    """
    try:
        import librosa
        import io

        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True, duration=120)

        features = {}

        # Duración
        features['duration_min'] = round(librosa.get_duration(y=y, sr=sr) / 60, 3)

        # Tempo (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = round(float(tempo), 1)

        # Loudness aproximada como RMS → convertir a dB
        rms = np.sqrt(np.mean(y**2))
        # Mapear RMS (0–1 aprox) a dB (-60 a 0)
        loudness_db = round(20 * np.log10(rms + 1e-9), 1)
        features['loudness'] = max(-60.0, min(0.0, loudness_db))

        # Tonalidad y Modo con Chroma features
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key = int(np.argmax(np.mean(chroma, axis=1)))
        features['key'] = key

        # Modo: comparar perfil mayor vs menor de la tonalidad detectada
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        chroma_mean = np.mean(chroma, axis=1)
        chroma_shifted = np.roll(chroma_mean, -key)
        major_score = np.corrcoef(chroma_shifted, major_profile)[0, 1]
        minor_score = np.corrcoef(chroma_shifted, minor_profile)[0, 1]
        features['mode'] = 1 if major_score > minor_score else 0

        # Instrumentalness aproximada: poca energía en MFCCs de timbre vocal → más instrumental
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        vocal_energy = np.mean(np.abs(mfccs[1:5]))  # coeficientes 1-4: zona de voz
        # Normalizar a 0-1 (inverso de presencia vocal)
        instrumentalness_est = max(0.0, min(1.0, 1.0 - float(vocal_energy) / 20.0))
        features['instrumentalness'] = round(instrumentalness_est, 3)

        return features, None

    except ImportError:
        return None, "librosa_not_installed"
    except Exception as e:
        return None, str(e)

# ─── UI PRINCIPAL ─────────────────────────────────────────────────────────────

st.markdown("# 🎙️ Audio Upload")
st.markdown("""
<div style='background:#fff3cd; padding:12px 18px; border-radius:8px; border-left: 4px solid #f0a500; margin-bottom:16px;'>
⚠️ <b>BETA</b> — Esta función está en desarrollo. Las features que Spotify calcula
con algoritmos propietarios (bailabilidad, valencia, energía...) no se pueden replicar
exactamente desde el audio crudo. Se extraen las que sí son posibles (tempo, duración,
tonalidad) y el resto los ajustas tú con sliders.
</div>
""", unsafe_allow_html=True)

reg_model, clf_model = load_models()
if reg_model is None:
    st.error("⚠️ No se pudieron cargar los modelos.")
    st.stop()

# Upload
uploaded_file = st.file_uploader(
    "Sube un archivo de audio (MP3 o WAV, máx. 50 MB):",
    type=['mp3', 'wav', 'ogg', 'm4a']
)

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format=uploaded_file.type)

    st.markdown("---")
    st.markdown("### 🔬 Extrayendo características de audio...")

    extracted, error = extract_features_librosa(audio_bytes)

    if error == "librosa_not_installed":
        st.error("""
        **librosa no está instalado.** Para activar la extracción automática:
        ```
        pip install librosa
        ```
        Por ahora, ajusta todas las features manualmente con los sliders de abajo.
        """)
        extracted = {}
    elif error:
        st.warning(f"No se pudieron extraer algunas features automáticamente: {error}")
        extracted = extracted or {}

    if extracted:
        st.markdown("#### ✅ Features extraídas automáticamente:")
        ext_cols = st.columns(len(extracted))
        for j, (k, v) in enumerate(extracted.items()):
            label_map = {
                'tempo': '🥁 Tempo (BPM)', 'loudness': '🔊 Volumen (dB)',
                'duration_min': '⏱️ Duración (min)', 'key': '🎹 Tonalidad',
                'mode': '🎵 Modo', 'instrumentalness': '🎹 Instrumental'
            }
            ext_cols[j].metric(label_map.get(k, k), str(v))

    st.markdown("---")
    st.markdown("#### 🎚️ Ajusta las features que requieren criterio humano:")

    manual_values = {}
    col1, col2 = st.columns(2)
    manual_feats = list(MANUAL_FEATURES.items())

    for i, (feat, (label, min_v, max_v, default, step)) in enumerate(manual_feats):
        with (col1 if i % 2 == 0 else col2):
            if isinstance(step, int):
                val = st.slider(label, int(min_v), int(max_v), int(default))
            else:
                val = st.slider(label, float(min_v), float(max_v), float(default), step=float(step))
            manual_values[feat] = val

    st.markdown("---")

    if st.button("🎯  Predecir con este audio", use_container_width=True, type="primary"):

        # Combinar features extraídas + manuales
        all_features = {'Unnamed: 0': 0}
        all_features.update(manual_values)
        all_features.update(extracted)  # las extraídas tienen prioridad

        # Predicción
        X = pd.DataFrame([{f: all_features.get(f, 0) for f in MODEL_FEATURES}])
        pop_pred  = float(np.clip(reg_model.predict(X)[0], 0, 100))
        hit_proba = float(clf_model.predict_proba(X)[0][1])

        st.markdown("## 📊 Resultado")
        r1, r2 = st.columns(2)

        with r1:
            color = "#ff2a36" if pop_pred >= 70 else "#f0a500" if pop_pred >= 50 else "#888"
            st.markdown(f"""
            <div style='text-align:center; padding:24px; background:#f9f6f0; border-radius:12px; border:2px solid {color}'>
                <div style='font-size:4rem; font-weight:900; color:{color}'>{pop_pred:.0f}</div>
                <div style='color:#888'>popularidad predicha (0–100)</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(pop_pred / 100)

        with r2:
            hit_color = "#ff2a36" if hit_proba >= 0.5 else "#333"
            st.markdown(f"""
            <div style='text-align:center; padding:24px; background:#f9f6f0; border-radius:12px; border:2px solid {hit_color}'>
                <div style='font-size:4rem; font-weight:900; color:{hit_color}'>{hit_proba*100:.1f}%</div>
                <div style='color:#888'>probabilidad de hit</div>
            </div>
            """, unsafe_allow_html=True)

            if hit_proba >= 0.5:
                st.success("🎯 ¡El modelo predice un HIT!")
            else:
                st.info("📊 El modelo predice que no será un hit.")

        st.caption("""
        ⚠️ **Recordatorio BETA:** La precisión de esta predicción depende de qué tan bien
        se hayan extraído las features y de los ajustes manuales. Los modelos se entrenaron
        con features calculadas por Spotify, no con extracción directa de audio.
        """)
