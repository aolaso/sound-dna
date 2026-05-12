"""
Página 3: What If
El usuario parte de una canción base (sliders) y luego modifica
una feature específica para ver cómo cambia la predicción.
También incluye un análisis de sensibilidad: ¿qué feature tiene más impacto?
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="What If | Sound DNA",
    page_icon="🔄",
    layout="wide"
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

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

# Cuánto varía cada feature en el análisis de sensibilidad
SENSITIVITY_DELTA = {
    'danceability': 0.2, 'energy': 0.2, 'key': 2, 'loudness': 5.0,
    'mode': 1, 'speechiness': 0.1, 'acousticness': 0.2,
    'instrumentalness': 0.2, 'liveness': 0.2, 'valence': 0.2,
    'tempo': 20, 'time_signature': 1, 'duration_min': 1.0,
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

def predict(reg_model, clf_model, feature_values):
    """Devuelve (popularidad_predicha, probabilidad_hit)"""
    X = build_input(feature_values)
    pop  = float(np.clip(reg_model.predict(X)[0], 0, 100))
    prob = float(clf_model.predict_proba(X)[0][1])
    return pop, prob

# ─── UI PRINCIPAL ─────────────────────────────────────────────────────────────

st.markdown("# 🔄 What If")
st.markdown("""
¿Qué pasaría si tu canción fuera más rápida? ¿Más acústica? ¿Más corta?

Define la canción base con los sliders, obtén su predicción y luego
explora cómo cambia al modificar una feature específica.
""")
st.markdown("---")

reg_model, clf_model = load_models()
if reg_model is None:
    st.error("⚠️ No se pudieron cargar los modelos. Verifica src/model/production/")
    st.stop()

# ─── PARTE 1: CANCIÓN BASE ────────────────────────────────────────────────────

st.markdown("## 1️⃣ Define tu canción base")

col1, col2, col3 = st.columns(3)
base_values = {'Unnamed: 0': 0}
cols_cycle = [col1, col2, col3]

for i, (feat, (label, min_v, max_v, default, step)) in enumerate(FEATURE_INFO.items()):
    with cols_cycle[i % 3]:
        if isinstance(step, int):
            val = st.slider(f"{label} (base)", int(min_v), int(max_v), int(default), key=f"base_{feat}")
        else:
            val = st.slider(f"{label} (base)", float(min_v), float(max_v), float(default), step=float(step), key=f"base_{feat}")
        base_values[feat] = val

st.markdown("---")

# ─── PARTE 2: MODIFICACIÓN ────────────────────────────────────────────────────

st.markdown("## 2️⃣ Modifica una feature y ve qué pasa")

feat_to_change = st.selectbox(
    "¿Qué feature quieres modificar?",
    options=list(FEATURE_INFO.keys()),
    format_func=lambda f: FEATURE_INFO[f][0]
)

label, min_v, max_v, _, step = FEATURE_INFO[feat_to_change]
current_val = base_values[feat_to_change]

if isinstance(step, int):
    new_val = st.slider(
        f"Nuevo valor de {label}",
        int(min_v), int(max_v), int(current_val),
        key="what_if_slider"
    )
else:
    new_val = st.slider(
        f"Nuevo valor de {label}",
        float(min_v), float(max_v), float(current_val),
        step=float(step), key="what_if_slider"
    )

if st.button("🔄  Calcular impacto", use_container_width=True, type="primary"):

    # Predicción base
    pop_base, prob_base = predict(reg_model, clf_model, base_values)

    # Predicción modificada
    modified_values = base_values.copy()
    modified_values[feat_to_change] = new_val
    pop_new, prob_new = predict(reg_model, clf_model, modified_values)

    delta_pop  = pop_new  - pop_base
    delta_prob = (prob_new - prob_base) * 100

    st.markdown("## 📊 Comparación: Base vs Modificada")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Canción base")
        st.metric("Popularidad", f"{pop_base:.0f}/100")
        st.metric("Prob. hit",   f"{prob_base*100:.1f}%")

    with c2:
        st.markdown("#### Canción modificada")
        st.metric("Popularidad", f"{pop_new:.0f}/100",
                  delta=f"{delta_pop:+.1f}")
        st.metric("Prob. hit",   f"{prob_new*100:.1f}%",
                  delta=f"{delta_prob:+.1f}%")

    with c3:
        st.markdown("#### Cambio aplicado")
        st.markdown(f"""
        <div style='background:#f9f6f0; padding:16px; border-radius:10px;'>
            <b>{label}</b><br>
            <span style='font-size:1.4rem'>{current_val} → {new_val}</span>
        </div>
        """, unsafe_allow_html=True)

        if abs(delta_pop) < 0.5:
            st.info("Este cambio tiene poco impacto en la popularidad.")
        elif delta_pop > 0:
            st.success(f"✅ Subir {label} mejora la popularidad en {delta_pop:.1f} puntos.")
        else:
            st.warning(f"⬇️ Subir {label} reduce la popularidad en {abs(delta_pop):.1f} puntos.")

    # ─── ANÁLISIS DE SENSIBILIDAD ─────────────────────────────────────────────

    st.markdown("---")
    st.markdown("## 🧪 Análisis de sensibilidad — ¿Qué feature tiene más impacto?")
    st.markdown("""
    Variamos cada feature ±una unidad típica desde los valores de tu canción base
    y medimos cuánto cambia la popularidad. Así sabemos qué knob conviene tocar.
    """)

    sensitivity_rows = []
    for feat in FEATURE_INFO:
        delta = SENSITIVITY_DELTA[feat]
        _, _, feat_max, _, _ = FEATURE_INFO[feat]

        # Impacto de subir la feature
        up_vals = base_values.copy()
        up_vals[feat] = min(base_values[feat] + delta, feat_max)
        pop_up, _ = predict(reg_model, clf_model, up_vals)

        # Impacto de bajar la feature
        _, feat_min, _, _, _ = FEATURE_INFO[feat][1], FEATURE_INFO[feat][1], FEATURE_INFO[feat][2], FEATURE_INFO[feat][3], FEATURE_INFO[feat][4]
        feat_min_val = FEATURE_INFO[feat][1]
        down_vals = base_values.copy()
        down_vals[feat] = max(base_values[feat] - delta, feat_min_val)
        pop_down, _ = predict(reg_model, clf_model, down_vals)

        impact = abs(pop_up - pop_base) + abs(pop_down - pop_base)
        sensitivity_rows.append({
            'Feature': FEATURE_INFO[feat][0],
            'Impacto total': round(impact, 2),
            '+delta': f"{pop_up - pop_base:+.1f}",
            '-delta': f"{pop_down - pop_base:+.1f}",
        })

    sens_df = pd.DataFrame(sensitivity_rows).sort_values('Impacto total', ascending=False)
    st.dataframe(sens_df.reset_index(drop=True), use_container_width=True)
    st.caption("**Impacto total** = suma del cambio de popularidad al subir y bajar la feature. Mayor = más influyente para esta canción concreta.")
