import joblib
import numpy as np

try:
    # Cargamos el árbol de decisión básico
    modelo_ia = joblib.load('mini_ia_hemoglobina.pkl')
except FileNotFoundError:
    print("Error: No se encontró 'mini_ia_hemoglobina.pkl'.")

# 🧠 MATRIZ DE DIAGNÓSTICO PREDICTIVO PROACTIVO (Basado en la proyección de la tendencia)
TEXTOS_PREDICTIVOS = {
    "recaida_anemia": "⚠️ Proyección Crítica (Walle-HB): Aunque tu estado actual está en rango, la velocidad de caída de tus niveles proyecta un alto riesgo de recaer en Anemia en tu próximo control si no intervienes.",
    "recaida_poliglobulia": "⚠️ Alerta de Alza Rápida (Walle-HB): Se detecta una curva empinada hacia el alza. El sistema proyecta una tendencia de riesgo hacia la Poliglobulia. Es urgente aumentar tu hidratación.",
    "mejora_sostenida": "📈 Tendencia Favorable (Walle-HB): La curva analítica muestra una estabilización progresiva y ascendente hacia rangos óptimos. Tu cuerpo consolida una excelente adaptación de transporte de oxígeno.",
    "homeostasis": "🔄 Homeostasis Predictiva: Walle-HB confirma que la variabilidad de tus datos históricos es mínima. Tu metabolismo mantiene un control idóneo y una estabilidad sostenida en el tiempo.",
    "primer_analisis": "📊 Historial inicial en proceso. A partir de tu siguiente control clínico, el motor estadístico de Walle-HB comenzará a trazar proyecciones y tendencias personalizadas."
}

# Recomendaciones predictivas adicionales basadas en la tendencia calculada
RECOMENDACIONES_TENDENCIA = {
    "riesgo_caida": [
        "• Modificar dieta proactivamente: Añadir legumbres y vegetales de hoja verde oscura para frenar la tendencia de descenso.",
        "• Planificar un control de ferritina preventivo antes de la fecha estándar de tu próximo análisis."
    ],
    "riesgo_alza": [
        "• Reducir drásticamente carnes rojas y alimentos muy ricos en hierro durante las siguientes 2 semanas.",
        "• Incrementar la ingesta de agua a 3 litros diarios para diluir preventivamente la concentración de glóbulos rojos."
    ],
    "estable": [
        "• Mantener el patrón actual de nutrición y los hábitos físicos que estabilizaron tu curva biológica."
    ]
}

RECOMENDACIONES_SALUD = {
    'Anemia': [
        "• Priorizar la ingesta de hierro hemínico (carnes magras, hígado) junto con jugos cítricos (Vitamina C) para acelerar la absorción.",
        "• Monitorear síntomas de fatiga, palidez o debilidad muscular durante tus actividades diarias en altitud."
    ],
    'Poliglobulia': [
        "• Incrementar de forma inmediata el consumo de agua líquida a 2.5 litros diarios para disminuir la viscosidad de la sangre.",
        "• Evitar totalmente los suplementos vitamínicos que contengan hierro o ácido fólico."
    ],
    'Estable': [
        "• Mantener tu patrón alimenticio actual para conservar tu capacidad de transporte de oxígeno ideal.",
        "• Continuar con actividad física moderada para aprovechar la excelente oxigenación actual de tus tejidos."
    ]
}

def generar_prediccion_y_recomendaciones(datos_paciente, historial_valores=None):
    """
    datos_paciente: [id_genero, altitud, edad, valor_min, valor_max, valor_hemoglobina]
    historial_valores: Lista dinámica desde C# ordenada [más_nuevo, ..., más_antiguo]
    """
    id_genero, altitud, edad, valor_min, valor_max, valor_actual = datos_paciente
    valor_actual = round(float(valor_actual), 2)

    # 1. El modelo clasifica el estado matemático en este instante exacto
    estado_predicho = modelo_ia.predict([[id_genero, altitud, edad, valor_min, valor_max, valor_actual]])[0]
    
    # 2. MOTOR DE ANÁLISIS PREDICTIVO (Evaluación de la curva completa)
    mensaje_evolutivo = TEXTOS_PREDICTIVOS["primer_analisis"]
    lista_recomendaciones = RECOMENDACIONES_SALUD.get(estado_predicho, RECOMENDACIONES_SALUD['Estable']).copy()
    
    # Necesitamos al menos 2 datos históricos reales para trazar una tendencia matemática
    if historial_valores and len(historial_valores) >= 2:
        # Invertimos el historial recibido para analizar la curva en orden cronológico correcto (pasado -> presente)
        valores_cronologicos = [round(float(v), 2) for v in reversed(historial_valores)]
        
        # Calculamos la pendiente (slope) de la curva usando regresión lineal simple sobre los índices
        indices = np.arange(len(valores_cronologicos))
        pendiente = np.polyfit(indices, valores_cronologicos, 1)[0]
        
        promedio_historico = np.mean(valores_cronologicos)
        variabilidad = np.std(valores_cronologicos) # Desviación estándar
        
        # 🧠 ESCENARIOS DE DIAGNÓSTICO PREDICTIVO AVANZADO
        if estado_predicho == 'Estable':
            # Escenario A: Está estable hoy, pero sus niveles vienen cayendo rápido en el historial
            if pendiente < -0.4:
                mensaje_evolutivo = TEXTOS_PREDICTIVOS["recaida_anemia"]
                lista_recomendaciones.extend(RECOMENDACIONES_TENDENCIA["riesgo_caida"])
            # Escenario B: Está estable hoy, pero la curva se dispara peligrosamente hacia arriba
            elif pendiente > 0.4:
                mensaje_evolutivo = TEXTOS_PREDICTIVOS["recaida_poliglobulia"]
                lista_recomendaciones.extend(RECOMENDACIONES_TENDENCIA["riesgo_alza"])
            # Escenario C: Si se mantiene plano dentro del rango saludable
            else:
                if variabilidad < 0.3:
                    mensaje_evolutivo = TEXTOS_PREDICTIVOS["homeostasis"]
                else:
                    mensaje_evolutivo = TEXTOS_PREDICTIVOS["mejora_sostenida"]
                lista_recomendaciones.extend(RECOMENDACIONES_TENDENCIA["estable"])
                
        elif estado_predicho == 'Anemia':
            if pendiente > 0.2:
                mensaje_evolutivo = "📈 Evolución Positiva: A pesar del rango actual de Anemia, el algoritmo detecta una recuperación sostenida de la curva en los últimos registros."
            else:
                mensaje_evolutivo = "📉 Alerta Estacionaria: La tendencia se mantiene estancada en niveles bajos. Se sugiere evaluar la adherencia al tratamiento."
                
        elif estado_predicho == 'Poliglobulia':
            if pendiente < -0.2:
                mensaje_evolutivo = "📉 Tendencia de Control: Los niveles están descendiendo paulatinamente hacia rangos seguros, mostrando efectividad en los hábitos de descompresión o hidratación."
            else:
                mensaje_evolutivo = "🔺 Curva Ascendente Crítica: La viscosidad sanguínea proyecta un aumento sostenido. Evitar esfuerzos extenuantes y consultar a tu médico."

    # 3. Ensamblaje modular del Reporte Inteligente
    intro_diagnostico = f"Reporte de Walle-HB: Tu estado actual es clasificado como {estado_predicho} con {valor_actual} g/dL."
    alerta_final = f"{intro_diagnostico}\n\n{mensaje_evolutivo}"
    
    return estado_predicho, [alerta_final], lista_recomendaciones
