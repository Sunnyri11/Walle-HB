import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# 1. Cargar tus datos reales
df = pd.read_csv('nivel_hemoglobina.csv')

# 2. Separar características y etiquetas
X = df[['id_genero', 'altitud', 'edad', 'valor_min', 'valor_max', 'valor_hemoglobina']]
y = df['estado_diagnostico']

# 3. Dividir el conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Configurar y entrenar el modelo
modelo_ia = DecisionTreeClassifier(max_depth=5, random_state=42)

# MODIFICACIÓN CLAVE: Usamos .values para entrenar con matrices numéricas puras.
# Esto evitará que la IA exija nombres de columnas exactos en FastAPI.
modelo_ia.fit(X_train.values, y_train)

# 5. Evaluar precisión usando también .values para consistencia
precision = modelo_ia.score(X_test.values, y_test)
print(f"¡Mini IA entrenada con éxito! Precisión del modelo: {precision * 100:.2f}%")

# 6. Guardar el archivo binario final
joblib.dump(modelo_ia, 'mini_ia_hemoglobina.pkl')
print("Modelo guardado con éxito como 'mini_ia_hemoglobina.pkl'")
