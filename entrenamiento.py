# filename: entrenamiento.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine, text

# Conexión Global a tu base de datos de Aiven
usuario_db = "avnadmin"
password_db = "AVNS_RAotqvpMbH5nDsAc7GR"
host_parte1 = "mysql-1e7e449b-sistemahb.g"
host_parte2 = ".aivencloud.com"
servidor_db = host_parte1 + host_parte2
puerto_db = "13585"
nombre_db = "sistemahb"
CONEXION_AIVEN = f"mysql+pymysql://{usuario_db}:{password_db}@{servidor_db}:{puerto_db}/{nombre_db}"

def obtener_engine():
    return create_engine(CONEXION_AIVEN, connect_args={"ssl": {"skip_verify": True}})

def calcular_etiqueta(row):
    valor = row["valor_hemoglobina"]
    v_min = row["valor_min"]
    v_max = row["valor_max"]
    if pd.isna(valor) or pd.isna(v_min) or pd.isna(v_max): return "Estable"
    if valor < v_min: return "Anemia"
    elif valor > v_max: return "Poliglobulia"
    else: return "Estable"

# =====================================================
# 📤 SUBIR EL PKL BINARIO A TU TABLA WALLE_MEMORY
# =====================================================
def guardar_pkl_en_aiven():
    if not os.path.exists("mini_ia_hemoglobina.pkl"):
        print("⚠️ No existe 'mini_ia_hemoglobina.pkl' local para subir.")
        return

    print("📤 Subiendo memoria binaria .pkl a la tabla walle_memory...")
    with open("mini_ia_hemoglobina.pkl", "rb") as f:
        bytes_binarios = f.read()

    engine = obtener_engine()
    with engine.begin() as conexion:
        # Limpiamos la tabla para que solo exista una memoria maestra activa
        conexion.execute(text("TRUNCATE TABLE walle_memory"))
        # Insertamos el archivo binario directamente en el campo 'memoria'
        conexion.execute(
            text("INSERT INTO walle_memory (memoria) VALUES (:bin)"),
            {"bin": bytes_binarios}
        )
    print("✅ ¡Memoria guardada en el campo 'memoria' de walle_memory exitosamente!")

# =====================================================
# 🌅 DESCARGAR EL PKL BINARIO AL DESPERTAR EN RENDER
# =====================================================
def cargar_pkl_desde_aiven():
    print("🌅 Descargando memoria binaria desde la tabla walle_memory...")
    try:
        engine = obtener_engine()
        with engine.connect() as conexion:
            resultado = conexion.execute(
                text("SELECT memoria FROM walle_memory LIMIT 1")
            ).fetchone()
            
            if resultado and resultado[0]:
                bytes_binarios = resultado[0]
                with open("mini_ia_hemoglobina.pkl", "wb") as f:
                    f.write(bytes_binarios)
                print("🔄 ¡Archivo 'mini_ia_hemoglobina.pkl' restaurado con éxito desde la base de datos!")
                return True
            else:
                print("ℹ️ Tabla 'walle_memory' vacía. Se procederá a entrenar el modelo inicial.")
                return False
    except Exception as e:
        print(f"⚠️ Nota al recuperar de walle_memory: {e}")
        return False

# =====================================================
# ENTRENAMIENTO COMPLETO
# =====================================================
def entrenar_modelo():
    try:
        print("🔌 Leyendo datos clínicos desde Aiven...")
        engine = obtener_engine()
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
        df_aiven = pd.read_sql(consulta_sql, engine)
        df_aiven.to_csv("nivel_hemoglobina.csv", index=False)
    except Exception:
        df_aiven = pd.DataFrame()

    try: df_local_historia = pd.read_csv("nivel_hemoglobina.csv")
    except Exception: df_local_historia = pd.DataFrame()

    try: df_sintetico = pd.read_csv("nivel_hemoglobina_train.csv")
    except Exception: df_sintetico = pd.DataFrame()

    df_reales_unidos = pd.concat([df_aiven, df_local_historia], ignore_index=True).drop_duplicates()
    
    if "estado_diagnostico" not in df_reales_unidos.columns:
        df_reales_unidos["estado_diagnostico"] = df_reales_unidos.apply(calcular_etiqueta, axis=1)
    if "estado_diagnostico" not in df_sintetico.columns:
        df_sintetico["estado_diagnostico"] = df_sintetico.apply(calcular_etiqueta, axis=1)

    df_real_ponderado = pd.concat([df_reales_unidos] * 100, ignore_index=True)
    df = pd.concat([df_real_ponderado, df_sintetico], ignore_index=True)

    columnas_clave = ["id_genero", "altitud", "edad", "valor_min", "valor_max", "valor_hemoglobina", "estado_diagnostico"]
    df = df.dropna(subset=columnas_clave)

    X = df[["id_genero", "altitud", "edad", "valor_min", "valor_max", "valor_hemoglobina"]]
    y = df["estado_diagnostico"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    modelo_ia = RandomForestClassifier(n_estimators=50, max_depth=None, random_state=42)
    modelo_ia.fit(X_train.values, y_train)

    joblib.dump(modelo_ia, "mini_ia_hemoglobina.pkl", compress=3)
    print("🧠 ¡Archivo .pkl local actualizado!")
    
    # 💥 SUBIDA AUTOMÁTICA AL TERMINAR
    guardar_pkl_en_aiven()
    
    return len(df_reales_unidos)

if __name__ == "__main__":
    entrenar_modelo()
