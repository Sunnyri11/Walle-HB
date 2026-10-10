import joblib

try:
    # Cargamos el árbol de decisión básico
    modelo_ia = joblib.load('mini_ia_hemoglobina.pkl')
except FileNotFoundError:
    print("Error: No se encontró 'mini_ia_hemoglobina.pkl'.")

# MATRIZ DE CONOCIMIENTO EVOLUTIVO: Bloques de texto que la IA aprenderá a combinar
TEXTOS_EVOLUTIVOS = {
    "mejora": "📈 ¡Excelente progreso! Walle-HB detecta que tus niveles están respondiendo favorablemente en comparación a tu histórico anterior. Tu esfuerzo en tu alimentación está dando resultados.",
    "empeoramiento": "📉 Alerta de tendencia (Walle-HB): El sistema detecta un declive o un cambio adverso en comparación a tus registros previos. Es sumamente importante corregir hábitos antes de tu próxima evaluación.",
    "estable_continuo": "🔄 Estabilidad sostenida: Walle-HB confirma que has logrado mantener tus niveles en equilibrio a lo largo del tiempo. Tu cuerpo se encuentra en un estado de homeostasis adecuado.",
    "primer_analisis": "📊 Primer registro analizado por Walle-HB. A partir de tu siguiente control, la IA comenzará a trazar tu curva evolutiva personalizada."
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
    historial_valores: Lista dinámica recibida desde C# con todo el historial [actual, anterior, penúltimo...]
    """
    id_genero, altitud, edad, valor_min, valor_max, valor_actual = datos_paciente

    # Aseguramos el redondeo limpio del valor actual enviado
    valor_actual = round(float(valor_actual), 2)

    # 1. El modelo clasifica el estado matemático actual ('Anemia', 'Poliglobulia', 'Estable')
    estado_predicho = modelo_ia.predict([[id_genero, altitud, edad, valor_min, valor_max, valor_actual]])[0]
    
    # 2. APRENDIZAJE DE TENDENCIA: Saltamos el valor actual [0] para evaluar el verdadero registro anterior [1]
    mensaje_evolutivo = TEXTOS_EVOLUTIVOS["primer_analisis"]
    
    if historial_valores and len(historial_valores) > 1:
        # El elemento [0] es la medición actual (13.8). El elemento [1] es la verdadera medición anterior.
        ultimo_valor_anterior = round(float(historial_valores[1]), 2)
        
        # Lógica de aprendizaje clínico basado en la evolución real del paciente
        if estado_predicho == 'Estable' and ultimo_valor_anterior < valor_min:
            mensaje_evolutivo = TEXTOS_EVOLUTIVOS["mejora"] # Salió de la anemia
        elif estado_predicho == 'Anemia' and ultimo_valor_anterior >= valor_min:
            mensaje_evolutivo = TEXTOS_EVOLUTIVOS["empeoramiento"] # Cayó en anemia
        elif estado_predicho == 'Poliglobulia' and ultimo_valor_anterior <= valor_max:
            mensaje_evolutivo = TEXTOS_EVOLUTIVOS["empeoramiento"] # Subió a rango crítico
        elif estado_predicho == 'Estable' and ultimo_valor_anterior > valor_max:
            mensaje_evolutivo = TEXTOS_EVOLUTIVOS["mejora"] # Bajó de la poliglobulia a estable
        elif estado_predicho == 'Estable' and (valor_min <= ultimo_valor_anterior <= valor_max):
            mensaje_evolutivo = TEXTOS_EVOLUTIVOS["estable_continuo"] # Se mantiene sano

    # 3. Ensamblaje modular y dinámico usando el parámetro real 'valor_actual' recibido
    intro_diagnostico = f"Reporte de Walle-HB: Tu estado actual es clasificado como {estado_predicho} con {valor_actual} g/dL."
    alerta_final = f"{intro_diagnostico}\n\n{mensaje_evolutivo}"
    
    # Extraemos las acciones médicas específicas para su salud actual
    acciones_sugeridas = RECOMENDACIONES_SALUD.get(estado_predicho, RECOMENDACIONES_SALUD['Estable'])
    
    return estado_predicho, [alerta_final], acciones_sugeridas
