from modelo import train_random_forest, X, y
import numpy as np
import matplotlib.pyplot as plt
import random

def max_depth_analysis(criterion:str, n_estimators:int):
    list_accuracy = list()
    repeat = 10
    init = 5
    final = 20
    list_depth = [n for n in range(init,final) for _ in range(repeat)]

    for i in  range(init,final): 
        # Entrenamiento y reporte del modelo
        for _ in range(repeat):
            dt, X_train, X_test, y_test, y_pred, accuracy, report = train_random_forest(X, y,criterion=criterion,n_estimators=n_estimators, max_depth=i, random_state=random.randint(1, 500))
            list_accuracy.append(accuracy)

    # Crear el gráfico de dispersión

    coeficientes = np.polyfit(list_depth, list_accuracy, (final-init)//2 ) 
    polinomio = np.poly1d(coeficientes)

    # Graficar puntos y curva de regresión
    x_line = np.linspace(min(list_depth), max(list_depth), 500)  # Para una línea suave
    y_line = polinomio(x_line)


    plt.scatter(list_depth, list_accuracy, color='blue', label='Puntos de datos', s=2)
    plt.plot(x_line, y_line, color='red', label=f'Regresión: {polinomio}',  linewidth=4)
    # Personalización
    plt.title(f"Resultado del Accuracy variando max_depth con {criterion} y n_estimators = {n_estimators} (Random Forest)")
    plt.xlabel("Máx profundidad del Árbol (Random Forest)")
    plt.ylabel("Accuracy")
    plt.legend()

    # Mostrar el gráfico
    plt.show()


dt, X_train, X_test, y_test, y_pred, accuracy, report = train_random_forest(X, y,criterion='gini',n_estimators=25, max_depth=8, random_state=120)

print(accuracy)
print(report)

#max_depth_analysis('gini', 100)
#max_depth_analysis('entropy', 100)

"""
# prueba 1 
max_depth_analysis('gini', 25)
max_depth_analysis('entropy', 25)

# prueba 2
max_depth_analysis('gini', 50)
max_depth_analysis('entropy', 50)

# prueba 3 
max_depth_analysis('gini', 75)
max_depth_analysis('entropy', 75)

# prueba 4 
max_depth_analysis('gini', 100)
max_depth_analysis('entropy', 100)
"""