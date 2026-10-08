# filename: main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from predictor import generar_prediccion_y_recomendaciones 

app = FastAPI(title="Walle-HB: IA Médica Evolutiva")

class DatosPaciente(BaseModel):
    id_genero: int
    altitud: float
    edad: int
    valor_min: float
    valor_max: float
    valor_hemoglobina: float
    historial_anteriores: Optional[List[float]] = [] # 📈 ¡Nueva lista dinámica recibida desde C#!

@app.post("/predecir")
def predecir_salud(paciente: DatosPaciente):
    lista_datos = [
        paciente.id_genero,
        paciente.altitud,
        paciente.edad,
        paciente.valor_min,
        paciente.valor_max,
        paciente.valor_hemoglobina
    ]
    
    # Le pasamos los datos actuales y el histórico para que la IA decida el mensaje óptimo
    estado, alertas, recomendaciones = generar_prediccion_y_recomendaciones(lista_datos, paciente.historial_anteriores)
    
    return {
        "estado": [str(estado)],
        "prediccion_ia": str(estado),
        "alertas": alertas,
        "recomendaciones": recomendaciones
    }
