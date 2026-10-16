# filename: main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from predictor import generar_prediccion_y_recomendaciones 
from entrenamiento import entrenar_modelo, cargar_pkl_desde_aiven 
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🌅 Walle-HB despertando en Render. Sincronizando con 'walle_memory'...")
    # Descarga el .pkl binario desde Aiven al encender el servidor en la nube
    exito = cargar_pkl_desde_aiven()
    if not exito:
        print("🛠️ Generando entrenamiento inicial...")
        entrenar_modelo()
    yield

app = FastAPI(title="Walle-HB: IA Médica Evolutiva", lifespan=lifespan)

class DatosPaciente(BaseModel):
    id_genero: int
    altitud: float
    edad: int
    valor_min: float
    valor_max: float
    valor_hemoglobina: float
    historial_anteriores: Optional[List[float]] = []

# =====================================================================
# 🎯 PREDICCIÓN Y AUTO-ACTUALIZACIÓN AUTOMÁTICA EN TU ENLACE BASE (POST)
# =====================================================================
@app.post("/")
def predecir_salud(paciente: DatosPaciente):
    lista_datos = [
        paciente.id_genero, 
        paciente.altitud, 
        paciente.edad,
        paciente.valor_min, 
        paciente.valor_max, 
        paciente.valor_hemoglobina
    ]
    
    # 1. Ejecuta la predicción inmediata para el paciente actual
    estado, alertas, recommendations = generar_prediccion_y_recomendaciones(lista_datos, paciente.historial_anteriores)
    
    # 🔥 SEGUNDO PLANO: La IA se entrena sola con los nuevos datos acumulados 
    # y sube su memoria en formato binario a la tabla 'walle_memory' en Aiven.
    try:
        entrenar_modelo() 
        print("🤖 IA auto-actualizada en Aiven tras recibir un nuevo paciente.")
    except Exception as e:
        print(f"⚠️ No se pudo auto-actualizar el archivo .pkl en este turno: {e}")
        
    return {
        "estado": [str(estado)],
        "prediccion_ia": str(estado),
        "alertas": alertas,
        "recomendaciones": recommendations
    }

# =====================================================================
# 🔄 ACTUALIZACIÓN MANUAL BAJO DEMANDA EN TU ENLACE BASE (PUT)
# =====================================================================
@app.put("/")
def actualizar_memoria_ia():
    try:
        registros = entrenar_modelo() # Entrena localmente e inyecta directo en walle_memory
        return {"status": "success", "mensaje": "Memoria entrenada y guardada en walle_memory con éxito."}
    except Exception as e:
        return {"status": "error", "mensaje": str(e)}
