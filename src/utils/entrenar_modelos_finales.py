"""
entrenar_modelos_finales.py
============================
Script todo-en-uno que entrena los dos modelos finales de Sound DNA
y los guarda COMPRIMIDOS con joblib para que pesen poco y entren en GitHub.

Lo que hace, paso a paso:
1. Carga el CSV original de Spotify (114k canciones).
2. Hace el preprocesamiento mínimo (filtrar popularity>=10, crear is_hit, duration_min).
3. Hace el train/test split.
4. Entrena un RandomForestRegressor para predecir popularity.
5. Entrena un RandomForestClassifier para predecir is_hit.
6. Guarda los dos modelos COMPRIMIDOS en src/model/production/.
7. Imprime el tamaño final de cada archivo y las métricas.

Los modelos se guardan como Pipelines (scaler + modelo en un solo archivo)
para que la app y el backend solo tengan que cargar UN archivo y ya pueden predecir.

CÓMO USARLO:
1. Mete este archivo en src/utils/ del repo.
2. Abre terminal en la raíz del repo (sound-dna/).
3. Ejecuta:
       python src/utils/entrenar_modelos_finales.py

Tarda entre 1 y 3 minutos según tu ordenador.
"""

import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, r2_score,
    f1_score, accuracy_score, recall_score, roc_auc_score,
    confusion_matrix,
)

# =============================================================================
# CONFIGURACIÓN — calculo las rutas relativas al sitio donde está este archivo
# =============================================================================
# Este script vive en src/utils/, así que la raíz del proyecto está dos niveles arriba
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent          # sound-dna/
SRC_DIR = SCRIPT_DIR.parent                       # sound-dna/src/

CSV_PATH = SRC_DIR / "data" / "raw" / "spotify_tracks.csv"
OUTPUT_DIR = SRC_DIR / "model" / "production"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGRESSION_PATH = OUTPUT_DIR / "modelo_regresion_final.pkl"
CLASSIFICATION_PATH = OUTPUT_DIR / "modelo_clasificacion_final.pkl"

# =============================================================================
# Hiperparámetros — bajos a propósito para que el .pkl pese poco
# =============================================================================
# n_estimators=50 (vs 100 del baseline): mitad de árboles → mitad de tamaño
# max_depth=15: limita la profundidad → árboles más compactos
# Esto sacrifica MUY poco las métricas a cambio de archivos pequeños
N_ESTIMATORS = 50
MAX_DEPTH = 15
N_JOBS = -1          # usar todos los cores del ordenador
RANDOM_STATE = 42
COMPRESSION = 3      # 0 = sin comprimir, 9 = máxima compresión. 3 es el sweet spot

# =============================================================================
# 1. CARGAR LOS DATOS
# =============================================================================
print("=" * 70)
print("ENTRENAMIENTO DE MODELOS FINALES — Sound DNA")
print("=" * 70)
print(f"\n[1/6] Cargando datos desde {CSV_PATH}...")

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"No encuentro el CSV en {CSV_PATH}.\n"
        f"Asegúrate de que el archivo spotify_tracks.csv está en src/data/raw/."
    )

df = pd.read_csv(CSV_PATH)
print(f"      Cargadas {len(df):,} canciones, {df.shape[1]} columnas.")

# =============================================================================
# 2. PREPROCESAMIENTO MÍNIMO
# =============================================================================
print("\n[2/6] Preprocesando...")

# 2.1 Quitar canciones con popularidad muy baja (<10) — son nichos extremos
n_antes = len(df)
df = df[df["popularity"] >= 10].copy()
print(f"      Filtradas {n_antes - len(df):,} canciones con popularity<10. "
      f"Quedan {len(df):,}.")

# 2.2 Quitar duplicados por track_id (canciones que aparecen en varios géneros)
if "track_id" in df.columns:
    n_antes = len(df)
    df = df.drop_duplicates(subset="track_id").reset_index(drop=True)
    print(f"      Eliminados {n_antes - len(df):,} duplicados por track_id. "
          f"Quedan {len(df):,}.")

# 2.3 Calcular duración en minutos
if "duration_ms" in df.columns:
    df["duration_min"] = df["duration_ms"] / 60000

# 2.4 Crear la columna is_hit (1 si popularity>=70, 0 si no)
df["is_hit"] = (df["popularity"] >= 70).astype(int)
print(f"      Hits (popularity>=70): {df['is_hit'].sum():,} "
      f"({df['is_hit'].mean()*100:.2f}% del dataset)")

# 2.5 Quitar nulos por seguridad
df = df.dropna()

# =============================================================================
# 3. SEPARAR FEATURES (X) Y TARGETS (y)
# =============================================================================
print("\n[3/6] Separando features y targets...")

# Columnas que NO son features (identificadores, texto, targets)
cols_no_features = [
    "track_id", "track_name", "artists", "album_name",
    "track_genre", "duration_ms",
    "popularity", "is_hit",
]
cols_no_features = [c for c in cols_no_features if c in df.columns]

X = df.drop(columns=cols_no_features)
y_reg = df["popularity"]
y_clf = df["is_hit"]

# Asegurar que todas las columnas son numéricas
X = X.select_dtypes(include=[np.number])

print(f"      Features finales ({X.shape[1]}): {list(X.columns)}")

# =============================================================================
# 4. TRAIN / TEST SPLIT
# =============================================================================
print("\n[4/6] Train/test split (80/20)...")

# Split único estratificado por is_hit (sirve para ambos targets)
X_train, X_test, idx_train, idx_test = train_test_split(
    X, df.index,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_clf,
)
y_train_reg = y_reg.loc[idx_train]
y_test_reg = y_reg.loc[idx_test]
y_train_clf = y_clf.loc[idx_train]
y_test_clf = y_clf.loc[idx_test]

print(f"      Train: {len(X_train):,}   Test: {len(X_test):,}")

# Guardar train/test a CSV (lo pide el profesor)
print("      Guardando train.csv y test.csv...")
train_csv = X_train.copy()
train_csv["popularity"] = y_train_reg.values
train_csv["is_hit"] = y_train_clf.values
train_csv.to_csv(SRC_DIR / "data" / "train.csv", index=False)

test_csv = X_test.copy()
test_csv["popularity"] = y_test_reg.values
test_csv["is_hit"] = y_test_clf.values
test_csv.to_csv(SRC_DIR / "data" / "test.csv", index=False)

# =============================================================================
# 5. ENTRENAR MODELO DE REGRESIÓN
# =============================================================================
print("\n[5/6] Entrenando RandomForestRegressor...")
inicio = time.time()

# Pipeline: scaler + modelo en un solo objeto
# Así, al cargar el pkl, ya tienes todo lo que necesitas para predecir
pipeline_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )),
])

pipeline_reg.fit(X_train, y_train_reg)
y_pred_reg = pipeline_reg.predict(X_test)

mae = mean_absolute_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"      Entrenado en {time.time()-inicio:.1f}s")
print(f"      MAE = {mae:.3f}    R² = {r2:.3f}")

# Guardar comprimido
print(f"      Guardando con joblib (compress={COMPRESSION})...")
joblib.dump(pipeline_reg, REGRESSION_PATH, compress=COMPRESSION)
size_reg_mb = REGRESSION_PATH.stat().st_size / (1024 * 1024)
print(f"      ✅ Guardado en {REGRESSION_PATH}")
print(f"      Tamaño final: {size_reg_mb:.2f} MB")

# =============================================================================
# 6. ENTRENAR MODELO DE CLASIFICACIÓN
# =============================================================================
print("\n[6/6] Entrenando RandomForestClassifier...")
inicio = time.time()

pipeline_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight="balanced",   # importante por el desbalanceo 5%/95%
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )),
])

pipeline_clf.fit(X_train, y_train_clf)
y_pred_clf = pipeline_clf.predict(X_test)
y_proba_clf = pipeline_clf.predict_proba(X_test)[:, 1]

f1 = f1_score(y_test_clf, y_pred_clf)
acc = accuracy_score(y_test_clf, y_pred_clf)
recall = recall_score(y_test_clf, y_pred_clf)
auc = roc_auc_score(y_test_clf, y_proba_clf)
cm = confusion_matrix(y_test_clf, y_pred_clf)
tn, fp, fn, tp = cm.ravel()

print(f"      Entrenado en {time.time()-inicio:.1f}s")
print(f"      F1 = {f1:.3f}    Accuracy = {acc:.3f}    Recall = {recall:.3f}    AUC = {auc:.3f}")
print(f"      Matriz: TN={tn} FP={fp} FN={fn} TP={tp}")

# Guardar comprimido
print(f"      Guardando con joblib (compress={COMPRESSION})...")
joblib.dump(pipeline_clf, CLASSIFICATION_PATH, compress=COMPRESSION)
size_clf_mb = CLASSIFICATION_PATH.stat().st_size / (1024 * 1024)
print(f"      ✅ Guardado en {CLASSIFICATION_PATH}")
print(f"      Tamaño final: {size_clf_mb:.2f} MB")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)
print(f"Regresión   ({size_reg_mb:5.2f} MB) — MAE={mae:.2f}  R²={r2:.3f}")
print(f"Clasificación ({size_clf_mb:5.2f} MB) — F1={f1:.3f}  Recall={recall:.3f}  AUC={auc:.3f}")

total = size_reg_mb + size_clf_mb
print(f"\nTotal: {total:.2f} MB")

if size_reg_mb < 100 and size_clf_mb < 100:
    print("\n✅ Ambos modelos pesan menos de 100MB → SE PUEDEN SUBIR A GITHUB.")
else:
    print("\n⚠️  Algún modelo sigue pesando más de 100MB.")
    print("   Edita este script y baja N_ESTIMATORS de 50 a 25, o MAX_DEPTH de 15 a 10.")

print("\nListo. Ya puedes hacer commit en GitHub Desktop.")
