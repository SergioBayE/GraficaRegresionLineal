import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Datos
x = np.array([23, 26, 30, 34, 43, 48, 52, 57, 58, 60, 64, 68])
y = np.array([651, 762, 854, 924, 1051, 1156, 1230, 1364, 1425, 1443, 1511, 1613])

# Calcular la regresión lineal
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Calcular Función de la regresión lineal
def predict(x):
    return slope * x + intercept

y_pred = predict(x)

# Crea la gráfica
plt.figure(figsize=(10, 6))

# Grafica los datos observados
plt.scatter(x, y, label='Datos observados', color='blue')

# Grafica la línea de regresión
plt.plot(x, y_pred, label=f'Línea de regresión: y = {slope:.2f}x + {intercept:.2f}', color='red')

# Muestra la línea de la pendiente desde el origen
plt.plot([0, max(x)], [intercept, predict(max(x))], label='Línea de la pendiente', color='purple', linestyle='--')

# Mostrar la línea vertical en el intercepto
plt.axvline(0, color='green', linestyle='--', label='Intersección')

# Añadir etiquetas para la pendiente y el intercepto
plt.text(5, predict(5), f'Pendiente: {slope:.2f}', fontsize=12, color='orange', verticalalignment='bottom')
plt.text(0, intercept, f'Intercepto: {intercept:.2f}', fontsize=12, color='green', verticalalignment='bottom')

# Configuraciones adicionales del gráfico
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Análisis de la Regresión Lineal')
plt.legend()
plt.grid(True)
plt.show()
