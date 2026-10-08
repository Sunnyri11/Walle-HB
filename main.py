# filename: main.py
from fastapi import FastAPI
from pydantic import BaseModel
# Importación corregida con el nombre real de la función
from predictor import generar_prediccion_y_recomendaciones 

# 1. Crear la instancia de la aplicación FastAPI
app = FastAPI(
    title="API de Predicción de Hemoglobina",
    description="API que predice Anemia, Poliglobulia o Estado Estable usando Machine Learning"
)

# 2. Definir el modelo de datos de entrada usando Pydantic
class DatosPaciente(BaseModel):
    id_genero: int           # Ejemplo: 1 para masculino, 2 para femenino
    altitud: float           # Ejemplo: 3640.00 (La Paz)
    edad: int                # Ejemplo: 29
    valor_min: float         # Ejemplo: 14.50
    valor_max: float         # Ejemplo: 18.50
    valor_hemoglobina: float # Ejemplo: 15.63

# 3. Crear la ruta POST para recibir los datos y devolver la predicción
@app.post("/predecir")
def predecir_salud(paciente: DatosPaciente):
    # Convertimos el objeto de Pydantic en la lista ordenada que espera tu IA
    lista_datos = [
        paciente.id_genero,
        paciente.altitud,
        paciente.edad,
        paciente.valor_min,
        paciente.valor_max,
        paciente.valor_hemoglobina
    ]
    
    # Llamamos a tu función de predictor.py
    estado, alertas, recomendaciones = generar_prediccion_y_recomendaciones(lista_datos)
    
    # Retornamos la respuesta estructurada en formato JSON
    return {
        "prediccion_ia": estado,
        "alertas": alertas,
        "recomendaciones": recomendaciones
    }
