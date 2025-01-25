import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Preparación de datos
def prepare_data(df, target_column, feature_columns):
    # Codificar variables categóricas
    le = LabelEncoder()
    X = df[feature_columns].copy()
    
    for column in X.select_dtypes(include=['object']).columns:
        X[column] = le.fit_transform(X[column].astype(str))
    
    y = df[target_column]
    if y.dtype == 'object':
        y = le.fit_transform(y.astype(str))
        for i in le.classes_:
            print(i)
    
    return X, y

# Entrenamiento y evaluación del modelo
def train_decision_tree(X, y, max_depth=5,criterion='gini', random_state=42):
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Crear y entrenar el modelo
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state, criterion= criterion)
    dt.fit(X_train, y_train)
    
    # Hacer predicciones
    y_pred = dt.predict(X_test)
    
    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=1)
    
    return dt, X_train, X_test, y_test, y_pred, accuracy, report

# Entrenamiento y evaluación del modelo
def train_random_forest(X, y, n_estimators=100, max_depth=None, criterion= 'gini',random_state=42):
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Crear y entrenar el modelo
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        criterion=criterion,
        bootstrap= True
    )
    rf.fit(X_train, y_train)
    
    # Hacer predicciones
    y_pred = rf.predict(X_test)
    
    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=1)
    
    return rf, X_train, X_test, y_test, y_pred, accuracy, report

# Visualizar el árbol
def plot_decision_tree(dt, feature_names, class_names=None, figsize=(20,10)):
    plt.figure(figsize=figsize)
    plot_tree(dt, feature_names=feature_names, class_names=class_names, 
              filled=True, rounded=True)
    plt.show()

target_column = "Descripción Presunto Responsable"
feature_columns = [ 'Año','Municipio', 'Departamento', 
                    'Región', 'Modalidad','Presunto Responsable',
                    'Abandono o Despojo Forzado de Tierras', 'Amenaza o Intimidación',
                    'Ataque Contra Misión Médica',
                    'Confinamiento o Restricción a la Movilidad', 'Desplazamiento Forzado',
                    'Extorsión', 'Lesionados Civiles', 'Pillaje', 'Tortura',
                    'Violencia Basada en Género', 'Otro Hecho Simultáneo',
                    'Total de Víctimas del Caso', 'Tipo de Armas']
df = pd.read_csv('preprocesado.csv')

# Preparar datos
X, y = prepare_data(df,target_column,feature_columns)
