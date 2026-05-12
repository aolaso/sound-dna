"""
Página 1: Test Your Idea
El usuario ajusta sliders con características de audio y obtiene:
- Popularidad predicha (regresión, 0-100)
- Probabilidad de ser un hit (clasificación, %)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Test Your Idea | Sound DNA",
    page_icon="🎸",
    layout="wide"
)

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

# Estas son las 14 features que usan los modelos, en el orden exacto del entrenamiento.
# "Unnamed: 0" era el índice del CSV original. Es un bug documentado:
# actúa como proxy de género, pero siempre se manda como 0 en la app.
MODEL_FEATURES = [
    'danceability', 'energy', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_min', 'is_short_track', 'energy_x_danceability',
    'acoustic_minus_energy', 'explicit_int',
    'macro_classical', 'macro_electronic', 'macro_folk_country',
    'macro_hiphop_rnb', 'macro_jazz_blues', 'macro_latin', 'macro_metal',
    'macro_mood_other', 'macro_pop', 'macro_reggae', 'macro_rock', 'macro_world',
    'key_0', 'key_1', 'key_2', 'key_3', 'key_4', 'key_5',
    'key_6', 'key_7', 'key_8', 'key_9', 'key_10', 'key_11',
    'ts_3', 'ts_4', 'ts_5',
]

def build_input(feature_values: dict):
    import pandas as pd
    key_val  = int(feature_values.get('key', 5))
    ts_val   = int(feature_values.get('time_signature', 4))
    dur      = float(feature_values.get('duration_min', 3.5))
    energy   = float(feature_values.get('energy', 0.65))
    dance    = float(feature_values.get('danceability', 0.60))
    acoustic = float(feature_values.get('acousticness', 0.20))

    row = {f: 0 for f in MODEL_FEATURES}
    for col in ['danceability','energy','loudness','mode','speechiness',
                'acousticness','instrumentalness','liveness','valence',
                'tempo','duration_min']:
        if col in feature_values:
            row[col] = feature_values[col]

    row['is_short_track']        = 1 if dur < 2.0 else 0
    row['energy_x_danceability'] = energy * dance
    row['acoustic_minus_energy'] = acoustic - energy
    row['explicit_int']          = 0
    row['macro_pop']             = 1

    key_col = f'key_{key_val}'
    if key_col in row: row[key_col] = 1

    ts_col = f'ts_{ts_val}'
    if ts_col in row: row[ts_col] = 1

    return pd.DataFrame([row])[MODEL_FEATURES]

# Información de cada feature para construir los sliders:
# (etiqueta, min, max, default, step, descripción)
FEATURE_INFO = {
    'danceability':     ('💃 Bailabilidad',      0.0,  1.0,  0.60, 0.01, 'Qué tan adecuada es la canción para bailar (ritmo, beat, pulso). 0=no bailable, 1=muy bailable'),
    'energy':           ('⚡ Energía',            0.0,  1.0,  0.65, 0.01, 'Intensidad y actividad percibidas. 0=relajada/suave, 1=muy intensa/eléctrica'),
    'key':              ('🎹 Tonalidad',          0,    11,   5,    1,    'Nota musical base. 0=Do, 1=Do#, 2=Re, 3=Re#, 4=Mi, 5=Fa, 6=Fa#, 7=Sol, 8=Sol#, 9=La, 10=La#, 11=Si'),
    'loudness':         ('🔊 Volumen (dB)',       -60,  0,    -7.0, 0.1,  'Volumen promedio en decibelios. -60=casi silencio, 0=máximo. Los hits suelen estar entre -8 y -4 dB'),
    'mode':             ('🎵 Modo',               0,    1,    1,    1,    '1=Mayor (sonido más alegre, luminoso) / 0=Menor (más oscuro, melancólico)'),
    'speechiness':      ('🎤 Voz hablada',        0.0,  1.0,  0.05, 0.01, 'Presencia de palabras habladas. <0.33=música, 0.33-0.66=mix rap/canción, >0.66=podcasts/spoken word'),
    'acousticness':     ('🎸 Acústica',           0.0,  1.0,  0.20, 0.01, 'Probabilidad de que sea grabación acústica (sin efectos digitales). 1=completamente acústica'),
    'instrumentalness': ('🎹 Instrumental',       0.0,  1.0,  0.05, 0.01, 'Ausencia de voz cantada. >0.5=probablemente instrumental. Los hits suelen tener valores bajos (tienen voz)'),
    'liveness':         ('🎭 En directo',         0.0,  1.0,  0.12, 0.01, 'Probabilidad de que sea grabación en vivo con público. >0.8=muy probable que sea live'),
    'valence':          ('😊 Valencia',           0.0,  1.0,  0.50, 0.01, 'Positividad emocional. 1=alegre/eufórica, 0=triste/oscura/agresiva'),
    'tempo':            ('🥁 Tempo (BPM)',        50,   220,  120,  1,    'Velocidad en beats por minuto. Pop/dance: 100-130 BPM. Hip-hop: 70-100 BPM'),
    'time_signature':   ('🎼 Compás',             3,    5,    4,    1,    'Pulsos por compás. 4=cuatro por cuatro (el más común en pop). 3=vals. 5=progresivo'),
    'duration_min':     ('⏱️ Duración (min)',     0.5,  8.0,  3.5,  0.1,  'Duración en minutos. Los hits actuales suelen durar entre 2.5 y 4 minutos'),
}

# ─── CARGA DE MODELOS ─────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    """
    Carga los dos modelos desde src/model/production/.
    Los detecta automáticamente por tamaño de archivo:
    - El más grande (~9 MB) = modelo de regresión
    - El más pequeño (~2 MB) = modelo de clasificación
    """
    model_dir = Path("src/model/production")

    if not model_dir.exists():
        return None, None, "❌ No se encontró la carpeta src/model/production/"

    joblib_files = sorted(
        model_dir.glob("*.joblib"),
        key=lambda f: f.stat().st_size,
        reverse=True  # el más grande primero = regresión
    )

    if len(joblib_files) < 2:
        return None, None, f"❌ Se necesitan 2 archivos .joblib, solo hay {len(joblib_files)}"

    try:
        reg_model = joblib.load(joblib_files[0])
        clf_model = joblib.load(joblib_files[1])
        return reg_model, clf_model, None
    except Exception as e:
        return None, None, f"❌ Error cargando modelos: {str(e)}"

# ─── UI PRINCIPAL ─────────────────────────────────────────────────────────────

st.markdown("# 🎸 Test Your Idea")
st.markdown("""
Ajusta los sliders para definir las características de audio de tu canción hipotética.
El modelo predice qué popularidad tendría y si llegaría a ser un hit.
""")
st.markdown("---")

reg_model, clf_model, error_msg = load_models()

if error_msg:
    st.error(error_msg)
    st.stop()

# ─── SLIDERS ──────────────────────────────────────────────────────────────────

col1, col2, col3 = st.columns(3)
feature_values = {}
cols_cycle = [col1, col2, col3]

for i, (feat, (label, min_v, max_v, default, step, desc)) in enumerate(FEATURE_INFO.items()):
    with cols_cycle[i % 3]:
        if isinstance(step, int):
            val = st.slider(label, int(min_v), int(max_v), int(default), step=1, help=desc)
        else:
            val = st.slider(label, float(min_v), float(max_v), float(default), step=float(step), help=desc)
        feature_values[feat] = val

st.markdown("---")

# ─── PREDICCIÓN ───────────────────────────────────────────────────────────────

if st.button("🔮  Predecir popularidad", use_container_width=True, type="primary"):

    # Construir el DataFrame con las features en el orden exacto del entrenamiento
    X = build_input(feature_values)

    # Regresión: predice un número de popularidad
    popularity_raw = float(reg_model.predict(X)[0])
    popularity_pred = max(0.0, min(100.0, popularity_raw))  # clamp 0–100

    # Clasificación: predice probabilidad de ser hit
    # predict_proba devuelve [prob_clase_0, prob_clase_1]
    # clase 1 = hit (popularidad ≥ 70)
    hit_proba = float(clf_model.predict_proba(X)[0][1])
    is_hit = hit_proba >= 0.5

    # ─── RESULTADO VISUAL ─────────────────────────────────────────────────────

    st.markdown("## 📊 Resultado de la predicción")

    res1, res2 = st.columns(2)

    with res1:
        st.markdown("### Popularidad predicha")
        color = "#ff2a36" if popularity_pred >= 70 else "#f0a500" if popularity_pred >= 50 else "#888"
        st.markdown(f"""
        <div style='text-align:center; padding:28px; background:#f9f6f0; border-radius:14px; border: 2px solid {color}'>
            <div style='font-size:5rem; font-weight:900; color:{color}; line-height:1'>{popularity_pred:.0f}</div>
            <div style='font-size:0.95rem; color:#888; margin-top:6px'>sobre 100 puntos</div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(popularity_pred / 100)

        if popularity_pred >= 70:
            st.success("✅ Zona de HIT — popularidad ≥ 70 (top 4.8% del dataset)")
        elif popularity_pred >= 50:
            st.warning("🟡 Popularidad media — entre 50 y 69")
        elif popularity_pred >= 30:
            st.info("🔵 Popularidad baja — entre 30 y 49")
        else:
            st.error("⬇️ Popularidad muy baja — menos de 30")

    with res2:
        st.markdown("### Probabilidad de ser hit")
        hit_color = "#ff2a36" if is_hit else "#333"
        st.markdown(f"""
        <div style='text-align:center; padding:28px; background:#f9f6f0; border-radius:14px; border: 2px solid {hit_color}'>
            <div style='font-size:5rem; font-weight:900; color:{hit_color}; line-height:1'>{hit_proba*100:.1f}%</div>
            <div style='font-size:0.95rem; color:#888; margin-top:6px'>probabilidad de hit</div>
        </div>
        """, unsafe_allow_html=True)

        if is_hit:
            st.success("🎯 **¡El modelo predice un HIT!** — probabilidad ≥ 50%")
        else:
            st.info(f"📊 El modelo predice que **no será un hit** (probabilidad {hit_proba*100:.1f}%)")

        st.caption("""
        **Contexto del modelo:** Solo el 4.8% de canciones son hits → las clases están muy desbalanceadas.
        El modelo tiene AUC=0.801 (bueno discriminando) pero F1=0.211 (precaución con los positivos).
        """)

    # ─── TOP 3 FEATURES ───────────────────────────────────────────────────────

    st.markdown("---")
    st.markdown("### 🔑 Las 3 features más importantes para el modelo")
    st.caption("Estas son las que más peso tienen en la decisión — hallazgo contraintuitivo del proyecto:")

    t1, t2, t3 = st.columns(3)

    with t1:
        v = feature_values['instrumentalness']
        badge = "✅ Bien" if v < 0.3 else "⚠️ Alto"
        st.metric("🎹 Instrumental", f"{v:.2f}", f"Importancia: 0.110")
        st.caption(f"{badge} — Los hits tienen voz humana. Valores < 0.3 son favorables.")

    with t2:
        v = feature_values['duration_min']
        badge = "✅ Bien" if 2.5 <= v <= 4.0 else "⚠️ Fuera del rango óptimo"
        st.metric("⏱️ Duración", f"{v:.1f} min", f"Importancia: 0.084")
        st.caption(f"{badge} — Los hits duran entre 2.5 y 4 min.")

    with t3:
        v = feature_values['speechiness']
        badge = "✅ Bien" if v < 0.2 else "⚠️ Muy hablado"
        st.metric("🎤 Voz hablada", f"{v:.2f}", f"Importancia: 0.084")
        st.caption(f"{badge} — Balance entre cantado y hablado. <0.2 es lo más común en hits.")
