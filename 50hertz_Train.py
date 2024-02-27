# Código para entrenar modelo de inteligencia artificial con Keras y TensorFlow,
# utilizando una red neuronal recurrente (RNN) con keras. La variable objetivo Potencia 
# promedio (TWh) generada por los parques aerogeneradores de la operadora (TSO) Alemana 50Hertz 
# a nivel nacional, calculada por Fecha/Date o diaria desde las 00:00:00 hrs hasta las 23:45:00 hrs
# con registros/intervalos cada 15 minutos, en un periodo de tiempo desde el 23/08/2019 al 22/09/2020,
# por lo tanto, para cada fila, estas columnas son las variables dependientes que se desea predecir.
# Como resultado, se muestra un Grafico de dispersión (X-axis = Indice de Fila o Fecha, Y-axis = 
# Potencia promedio por Fecha) de la Potencia promedio Real y Potencia predicha. También,se muestra 
# una tabla de resultados con las columnas Date, Real_Promedio, Predicho_Promedio, Diferencia_Absoluta_Promedio 
# entre las dos columnas anteriores e Indice de Fila o Fecha.


# Importar las bibliotecas necesarias - última actualización de este código  14/02/2024.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
import matplotlib.pyplot as plt
import joblib
import time


data_path = '/home/valiente/aiValentin/windPower/50Hertz (copy).csv'
df = pd.read_csv(data_path, dayfirst=True)  # Agregar dayfirst=True para el formato de fecha
required_columns = ['Date'] + [f'{i:02d}:00:00' for i in range(24)]
missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(f"Faltan las siguientes columnas requeridas: {missing_columns}")

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)


# Dividir el conjunto de datos en características (X) y etiquetas (y)
X = df.drop(['Date'], axis=1)
y = df['00:00:00']  # Seleccionar la columna de la hora que desees para 'Real'

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el modelo de regresión RandomForest, el cual es un tipo de modelo basado en
# el algoritmo de Bosques Aleatorios (Random Forest), utilizado para el aprendizaje automático,
# tanto en problemos de Regresión como de Clasificación.

model = RandomForestRegressor(random_state=42)

print("Antes de entrenar el modelo")

# Entrenar el Modelo de Regresión lineal Multiple, con varias variables independientes.
model.fit(X_train, y_train)

print("Después de entrenar el modelo")

# Realizar predicciones en el conjunto de prueba aplicando el Modelo de Regresión lineal Múltiple.  
predictions = model.predict(X_test)

# Calcular el error cuadrático medio (Mean Squared Error - MSE) entre las medias, para evaluar la  
# calidad de un modelo de regresión, mientras más bajo, mejor su desempeño, no existe un valor standarde comparación.

mse_promedio = mean_squared_error(y, model.predict(X))
print(f"\nError cuadrático medio para potencias promedio por fecha: {mse_promedio:.2f}")

# Crear una tabla de resultados
results = pd.DataFrame({'Indice de Fila': y_test.index, 'Real': df.loc[y_test.index, '00:00:00'], 'Predicho': predictions, 'Diferencia Absoluta': abs(y_test - predictions)})
results = results.sort_values(by='Indice de Fila').reset_index(drop=True)
results['Real'] = results['Real'].apply(lambda x: f'{x:.2f}')
results['Predicho'] = results['Predicho'].apply(lambda x: f'{x:.2f}')
results['Diferencia Absoluta'] = results['Diferencia Absoluta'].apply(lambda x: f'{x:.2f}')

# Imprimir la tabla de resultados
#print("\nTabla de resultados - Potencia promedio global nacional por fecha en Teravatios (TWh)- Operadora 50Hertz:")
#print(tabulate(results, headers='keys', tablefmt='pretty', showindex=False))

# Calcular potencias promedio por fecha e intervalos de 15 minutos, desde 00:00:00 hrs hasta 23:45:00 hrs.
df['Real_Promedio'] = df.drop('Date', axis=1).mean(axis=1)
df['Predicho_Promedio'] = model.predict(X)
df['Diferencia_Absoluta_Promedio'] = abs(df['Real_Promedio'] - df['Predicho_Promedio'])
results_promedio = df[['Date', 'Real_Promedio', 'Predicho_Promedio', 'Diferencia_Absoluta_Promedio']].copy()
results_promedio['Indice de Fila'] = df.index
results_promedio = results_promedio.drop_duplicates().reset_index(drop=True)
results_promedio['Real_Promedio'] = results_promedio['Real_Promedio'].apply(lambda x: f'{x:.2f}')
results_promedio['Predicho_Promedio'] = results_promedio['Predicho_Promedio'].apply(lambda x: f'{x:.2f}')
results_promedio['Diferencia_Absoluta_Promedio'] = results_promedio['Diferencia_Absoluta_Promedio'].apply(lambda x: f'{x:.2f}')
print("\nTabla de resultados - Potencia Promedio Global Nacional por fecha en Teravatios (TWh)- Operadora Alemana 50Hertz:")
print(tabulate(results_promedio, headers='keys', tablefmt='pretty', showindex=False))

# Graficar resultados
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Reales', marker='o')
plt.plot(y_test.index, predictions, label='Predichos', marker='o')
plt.title('Potencia Promedio Global Nacional por Fecha en Teravatios (TWh): Reales Vs Predichos - Operadora Alemana 50Hertz')
plt.xlabel('Indice de Fila o Fecha: periodo 23/08/2019 al 22/09/2020 - Intervalos diarios cada 15 minutos')
plt.ylabel('Potencia promedio por fecha (TWh)')
plt.legend()
plt.show()

# Mostrar la tabla de resultados en el gráfico
print("Antes de mostrar la tabla de resultados")
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')
ax.table(cellText=results_promedio.values, colLabels=results_promedio.columns, cellLoc='center', loc='center')
plt.show()
print("Después de mostrar la tabla de resultados")


# Mostrar información adicional para verificar el flujo del script
print("Después de mostrar la tabla de resultados")

# Grafico de dispersión
plt.figure(figsize=(8, 8))
plt.scatter(y_test, predictions)
plt.title('Grafico de dispersión: Potencia Promedio Global Nacional por fecha en Teravatios (TWh): Real Vs Predicha - Operadora Alemana 50Hertz')
plt.xlabel('Potencia Promedio Real')
plt.ylabel('Potencia promedio Predicha')
plt.show()

# Guardar el modelo entrenado en un archivo con extensión ".pkl", es decir en formato "pickle", que  
# es un formato de serialización de Python y utiliza la función 'joblib.dump' para almacenar objetos 
# Python de manera más eficiente.

model_path = '/home/valiente/aiValentin/windPower/50Hertz_Modelo_entrenado.pkl'
joblib.dump(model, model_path)
print(f"Modelo guardado en: {model_path}")

# Cargar el modelo para verificar
loaded_model = joblib.load(model_path)
print("Modelo cargado exitosamente.")

# Imprimir mensaje final
print("Fin del script")

# Agregar un pequeño retraso para asegurarse de que las visualizaciones se completen
time.sleep(2)
