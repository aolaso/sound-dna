import streamlit as st

st.set_page_config(
    page_title="Sound DNA",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #111; }
    .main-header { color: #ff2a36; font-size: 3.2rem; font-weight: 900; line-height: 1.1; }
    .main-sub { color: #555; font-size: 1.25rem; font-style: italic; margin-bottom: 2rem; }
    .card { background: #f9f6f0; border-radius: 12px; padding: 20px; margin: 8px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🧬 Sound DNA</h1>', unsafe_allow_html=True)
st.markdown('<p class="main-sub">¿Puede el sonido predecir el éxito de una canción?</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("🎵 Canciones analizadas", "114,000")
col2.metric("🎼 Géneros distintos", "114")
col3.metric("📊 R² regresión", "0.613")
col4.metric("🎯 AUC clasificación", "0.801")

st.markdown("---")

st.markdown("""
## ¿Qué hace Sound DNA?

Analiza las **características de audio** de una canción — energía, tempo, bailabilidad,
si tiene voz humana, cuánto dura — y predice dos cosas a la vez:

| Modelo | Pregunta que responde | Resultado |
|--------|-----------------------|-----------|
| 🔢 **Regresión** | ¿Qué puntuación de popularidad tendrá? | Número del 0 al 100 |
| 🎯 **Clasificación** | ¿Será un hit? | Sí / No + probabilidad |

> **¿Qué es un hit?** Una canción con popularidad ≥ 70 en Spotify. Solo el 4.8% del dataset
> llega ahí. Es el top de lo más escuchado.

---

## Los 4 modos de la app

Usa el menú de la izquierda 👈

- 🎸 **Test Your Idea** — Diseña una canción hipotética con sliders y descubre su potencial
- 🔍 **Find Similar** — Encuentra canciones parecidas en el dataset real de 114k tracks
- 🔄 **What If** — ¿Qué pasa si subo la energía? ¿Si la hago más corta? Explora cómo cambia la predicción
- 🎙️ **Audio Upload BETA** — Sube un MP3 o WAV y extrae sus características automáticamente

---

### 🔑 Hallazgo más importante del proyecto

Contrariamente a la intuición, las features que más predicen el éxito **no son** energía
ni bailabilidad. Son:

1. **Instrumentalness** (0.110) — Los hits tienen voz humana. Las canciones muy instrumentales no suelen ser hits.
2. **Duration_min** (0.084) — Las canciones más cortas rinden mejor.
3. **Speechiness** (0.084) — Un balance óptimo entre hablado y cantado.

---

*Proyecto Final ML · The Bridge Data Science Bootcamp 2026*
*RandomForestRegressor + RandomForestClassifier · Pipeline StandardScaler + modelo*
""")
