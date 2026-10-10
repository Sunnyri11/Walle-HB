import joblib
 
modelo = joblib.load("mini_ia_hemoglobina.pkl")
 
print(modelo.predict([[1,3640,25,14,18,10]]))
print(modelo.predict([[1,3640,25,14,18,16]]))
print(modelo.predict([[1,3640,25,14,18,20]]))
