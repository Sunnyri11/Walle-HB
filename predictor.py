import joblib

# Cargar la IA entrenada
modelo_ia = joblib.load('mini_ia_hemoglobina.pkl')

def generar_prediccion_y_recomendaciones(datos_paciente):
    """
    datos_paciente debe ser una lista con la estructura:
    [id_genero, altitud, edad, valor_min, valor_max, valor_hemoglobina]
    Ejemplo para Wilder en La Paz: [1, 3640.00, 29, 14.50, 18.50, 15.63]
    """
    # 1. La IA predice el estado
    estado_predicho = modelo_ia.predict([datos_paciente])[0]
    
    # 2. Sistema de Alertas Dinámicas y Recomendaciones
    alertas = []
    recomendaciones = []
    
    if estado_predicho == 'Anemia':
        alertas.append("⚠️ ALERTA: Niveles de hemoglobina por debajo del rango recomendado.")
        recomendaciones.append("• Incorporar alimentos ricos en hierro (carnes magras, legumbres, espinacas).")
        recommendaciones.append("• Consumir jugos cítricos (Vitamina C) junto a tus comidas para mejorar la absorción.")
        recomendaciones.append("• Consultar a un médico para evaluar un perfil de hierro completo.")
        
    elif estado_predicho == 'Poliglobulia':
        alertas.append("🚨 ALERTA CRÍTICA: Niveles elevados (Posible Poliglobulia por Altitud).")
        recomendaciones.append("• Mantener una hidratación constante (mínimo 2 a 2.5 litros de agua al día).")
        recomendaciones.append("• Evitar el tabaco y ambientes con toxinas que reduzcan el oxígeno en sangre.")
        recomendaciones.append("• Programar una cita con un hematólogo para valorar una flebotomía preventiva.")
        
    else: # Estado Estable
        alertas.append(f"✅ ESTADO: ESTABLE ({datos_paciente[5]} G/DL)")
        recomendaciones.append("• Tus niveles se encuentran perfectamente adaptados a la altitud de tu ciudad.")
        recomendaciones.append("• Continúa con tu dieta equilibrada y estilo de vida activo.")
        recomendaciones.append("• Realiza un análisis clínico de control en 6 meses.")
        
    return estado_predicho, alertas, recomendaciones

# --- PRUEBA EN VIVO CON LOS DATOS DE WILDER ---
datos_wilder = [1, 3640.0, 29, 14.5, 18.5, 15.63]
estado, lista_alertas, lista_recom = generar_prediccion_y_recomendaciones(datos_wilder)

print(f"Predicción de la IA: {estado}")
print(f"Alertas en Pantalla: {lista_alertas}")
print("Recomendaciones sugeridas:")
for r in lista_recom:
    print(r)
