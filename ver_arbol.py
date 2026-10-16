# filename: ver_arbol.py
import joblib
from sklearn.tree import export_text

modelo = joblib.load("mini_ia_hemoglobina.pkl")

columnas = ["id_genero", "altitud", "edad", "valor_min", "valor_max", "valor_hemoglobina"]
lineas_totales_bosque = 0

# Recorremos cada uno de los 50 árboles del bosque y sumamos sus líneas lógicas
for i, arbol_individual in enumerate(modelo.estimators_):
    reglas_texto = export_text(arbol_individual, feature_names=columnas)
    lineas_totales_bosque += reglas_texto.count('\n')

print(f"📊 REPORTE DE DENSIDAD MASIVA (Walle-HB - Random Forest) 📊")
print(f"--------------------------------------------------")
print(f"• Número de árboles entrenados en paralelo: {len(modelo.estimators_)}")
print(f"• LÍNEAS LÓGICAS DE MEMORIA ACUMULADAS EN EL PKL: {lineas_totales_bosque}")
print(f"--------------------------------------------------\n")
