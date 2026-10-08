# filename: ia de hemoglobina.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# 1. Cargar tus datos reales extraídos de la Base de Datos
df = pd.read_csv('nivel_hemoglobina.csv')

# DINÁMICO: Aseguramos que la columna 'estado_diagnostico' exista calculándola 
# dinámicamente fila por fila para cualquier combinación de rangos y valores.
def calcular_etiqueta(row):
    valor = row['valor_hemoglobina']
    v_min = row['valor_min']
    v_max = row['valor_max']
    if valor < v_min:
        return 'Anemia'
    elif valor > v_max:
        return 'Poliglobulia'
    else:
        return 'Estable'

if 'estado_diagnostico' not in df.columns:
    df['estado_diagnostico'] = df.apply(calcular_etiqueta, axis=1)

# 2. Separar características (Variables de entrada del paciente) y etiquetas (Diagnóstico)
X = df[['id_genero', 'altitud', 'edad', 'valor_min', 'valor_max', 'valor_hemoglobina']]
y = df['estado_diagnostico']

# 3. Dividir el conjunto de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Configurar y entrenar el modelo (Árbol de Decisión)
# Mantenemos un max_depth adecuado para evitar sobreajuste (overfitting)
modelo_ia = DecisionTreeClassifier(max_depth=5, random_state=42)

# MODIFICACIÓN CLAVE: Usamos .values para entrenar con matrices numéricas puras.
# Esto evitará que la IA exija nombres de columnas exactos en FastAPI al recibir JSONs.
modelo_ia.fit(X_train.values, y_train)

# 5. Evaluar precisión usando también .values para consistencia
precision = modelo_ia.score(X_test.values, y_test)
print(f"¡Mini IA entrenada con éxito! Precisión del modelo: {precision * 100:.2f}%")

# 6. Guardar el archivo binario final que usará el predictor
joblib.dump(modelo_ia, 'mini_ia_hemoglobina.pkl')
print("Modelo guardado con éxito como 'mini_ia_hemoglobina.pkl'")
