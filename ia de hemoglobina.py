# filename: ia_de_hemoglobina.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# =====================================================
# FUNCIÓN PARA GENERAR ETIQUETAS SI NO EXISTEN
# =====================================================

def calcular_etiqueta(row):

    valor = row["valor_hemoglobina"]
    v_min = row["valor_min"]
    v_max = row["valor_max"]

    if valor < v_min:
        return "Anemia"
    elif valor > v_max:
        return "Poliglobulia"
    else:
        return "Estable"


# =====================================================
# CARGAR CSV REAL
# =====================================================

df_real = pd.read_csv("nivel_hemoglobina.csv")

if "estado_diagnostico" not in df_real.columns:
    df_real["estado_diagnostico"] = df_real.apply(
        calcular_etiqueta,
        axis=1
    )

# =====================================================
# CARGAR CSV SINTÉTICO
# =====================================================

df_sintetico = pd.read_csv("nivel_hemoglobina_train.csv")

if "estado_diagnostico" not in df_sintetico.columns:
    df_sintetico["estado_diagnostico"] = df_sintetico.apply(
        calcular_etiqueta,
        axis=1
    )

# =====================================================
# DAR MÁS PESO A LOS DATOS REALES
# =====================================================

df_real_ponderado = pd.concat(
    [df_real] * 10,
    ignore_index=True
)

# =====================================================
# UNIR AMBOS DATASETS
# =====================================================

df = pd.concat(
    [df_real_ponderado, df_sintetico],
    ignore_index=True
)

print("\n=== DISTRIBUCIÓN DEL DATASET ===")
print(df["estado_diagnostico"].value_counts())

# =====================================================
# VARIABLES DE ENTRADA
# =====================================================

X = df[
    [
        "id_genero",
        "altitud",
        "edad",
        "valor_min",
        "valor_max",
        "valor_hemoglobina"
    ]
]

y = df["estado_diagnostico"]

# =====================================================
# DIVISIÓN ENTRENAMIENTO / PRUEBA
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# MODELO
# =====================================================

modelo_ia = DecisionTreeClassifier(
    max_depth=6,
    random_state=42
)

modelo_ia.fit(
    X_train.values,
    y_train
)

# =====================================================
# MÉTRICAS
# =====================================================

precision = modelo_ia.score(
    X_test.values,
    y_test
)

print("\n=== RESULTADOS ===")
print(f"Precisión: {precision * 100:.2f}%")

predicciones = modelo_ia.predict(
    X_test.values
)

print("\n=== REPORTE ===")
print(
    classification_report(
        y_test,
        predicciones
    )
)

# =====================================================
# IMPORTANCIA DE VARIABLES
# =====================================================

columnas = [
    "id_genero",
    "altitud",
    "edad",
    "valor_min",
    "valor_max",
    "valor_hemoglobina"
]

print("\n=== IMPORTANCIA DE VARIABLES ===")

for nombre, importancia in zip(
    columnas,
    modelo_ia.feature_importances_
):
    print(
        f"{nombre}: {importancia:.4f}"
    )

# =====================================================
# GUARDAR MODELO
# =====================================================

joblib.dump(
    modelo_ia,
    "mini_ia_hemoglobina.pkl"
)

print(
    "\nModelo guardado como "
    "'mini_ia_hemoglobina.pkl'"
)
