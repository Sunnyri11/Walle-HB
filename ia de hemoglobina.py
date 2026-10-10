import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sqlalchemy import create_engine

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
# EXTRAER DATOS EN VIVO DESDE AIVEN
# =====================================================
usuario_db = "avnadmin"
password_db = "AVNS_RAotqvpMbH5nDsAc7GR"

# EVITAMOS EL RECORTE: Unimos el host en dos partes de texto plano
host_parte1 = "mysql-1e7e449b-sistemahb.g"
host_parte2 = ".aivencloud.com"
servidor_db = host_parte1 + host_parte2

puerto_db = "13585"
nombre_db = "sistemahb"

# Juntamos los componentes en el formato oficial requerido por SQLAlchemy (mysql+pymysql)
CONEXION_AIVEN = f"mysql+pymysql://{usuario_db}:{password_db}@{servidor_db}:{puerto_db}/{nombre_db}"

try:
    print("🔌 Conectando a la base de datos en la nube de Aiven...")
    engine = create_engine(
        CONEXION_AIVEN,
        connect_args={"ssl": {"skip_verify": True}}
    )
    
    # 🧠 CORREGIDO: Eliminados los paréntesis erróneos en 'fn.fecha_de_nacimiento'
    consulta_sql = """
        SELECT 
            p.id_genero AS id_genero,
            c.altitud AS altitud,
            (YEAR(CURDATE()) - YEAR(fn.fecha_de_nacimiento)) - (RIGHT(CURDATE(), 5) < RIGHT(fn.fecha_de_nacimiento, 5)) AS edad,
            rh.valor_min AS valor_min,
            rh.valor_max AS valor_max,
            nh.valor_hemoglobina AS valor_hemoglobina
        FROM nivel_hemoglobina nh
        INNER JOIN usuario u ON nh.id_usuario = u.id_usuario
        INNER JOIN persona p ON u.id_persona = p.id_persona
        INNER JOIN ciudad c ON p.id_ciudad = c.id_ciudad
        INNER JOIN rango_hemoglobina rh ON nh.id_rango_hemoglobina = rh.id_rango
        INNER JOIN fecha_nacimiento fn ON p.id_fecha_nacimiento = fn.id_fecha_nacimiento
    """
    
    # Descargamos los datos clínicos reales y sobreescribimos el CSV local
    df_real = pd.read_sql(consulta_sql, engine)
    print(f"✅ ¡Datos en vivo descargados con éxito! Se obtuvieron {len(df_real)} registros reales.")
    df_real.to_csv("nivel_hemoglobina.csv", index=False)

except Exception as e:
    print(f"⚠️ No se pudo conectar a Aiven ({e}). Se usará el archivo 'nivel_hemoglobina.csv' local existente.")
    df_real = pd.read_csv("nivel_hemoglobina.csv")

# Generamos etiquetas para los datos reales si faltan
if "estado_diagnostico" not in df_real.columns:
    df_real["estado_diagnostico"] = df_real.apply(calcular_etiqueta, axis=1)

# =====================================================
# CARGAR CSV SINTÉTICO
# =====================================================
df_sintetico = pd.read_csv("nivel_hemoglobina_train.csv")

if "estado_diagnostico" not in df_sintetico.columns:
    df_sintetico["estado_diagnostico"] = df_sintetico.apply(calcular_etiqueta, axis=1)

# =====================================================
# DAR MÁS PESO A LOS DATOS REALES (PONDERACIÓN X10)
# =====================================================
df_real_ponderado = pd.concat([df_real] * 10, ignore_index=True)

# =====================================================
# UNIR AMBOS DATASETS
# =====================================================
df = pd.concat([df_real_ponderado, df_sintetico], ignore_index=True)

print("\n=== DISTRIBUCIÓN DEL DATASET ===")
print(df["estado_diagnostico"].value_counts())

# =====================================================
# VARIABLES DE ENTRADA
# =====================================================
X = df[["id_genero", "altitud", "edad", "valor_min", "valor_max", "valor_hemoglobina"]]
y = df["estado_diagnostico"]

# =====================================================
# DIVISIÓN ENTRENAMIENTO / PRUEBA
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# =====================================================
# MODELO
# =====================================================
modelo_ia = DecisionTreeClassifier(max_depth=6, random_state=42)
modelo_ia.fit(X_train.values, y_train)

# =====================================================
# MÉTRICAS
# =====================================================
precision = modelo_ia.score(X_test.values, y_test)
print("\n=== RESULTADOS ===")
print(f"Precisión: {precision * 100:.2f}%")

predicciones = modelo_ia.predict(X_test.values)
print("\n=== REPORTE ===")
print(classification_report(y_test, predicciones))

# =====================================================
# IMPORTANCIA DE VARIABLES
# =====================================================
columnas = ["id_genero", "altitud", "edad", "valor_min", "valor_max", "valor_hemoglobina"]
print("\n=== IMPORTANCIA DE VARIABLES ===")
for nombre, importancia in zip(columnas, modelo_ia.feature_importances_):
    print(f"{nombre}: {importancia:.4f}")

# =====================================================
# GUARDAR MODELO
# =====================================================
joblib.dump(modelo_ia, "mini_ia_hemoglobina.pkl")
print("\nModelo guardado como 'mini_ia_hemoglobina.pkl'")
