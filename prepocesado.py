import pandas as pd
import numpy as np

df = pd.read_csv("datos.csv")
df.set_index('ID Caso', inplace=True)

# Obtener los valores únicos de una columna específica
def valores_unicos_df(dataframe, columna):
    if columna not in dataframe.columns:
        print(f"La columna '{columna}' no existe en el DataFrame.")
        return
    # Obtener valores únicos sin nulos
    valores_unicos = dataframe[columna].dropna().unique()
    conteos = df[columna].value_counts(dropna=False)
    print(f"Valores únicos en la columna '{columna}':")
    for valor in valores_unicos:
        print(f"{valor} / {conteos[valor]}")


def preprocesar_dataSet(df):
    del df["ID Caso Relacionado"]
    del df["Código DANE de Municipio"]
    del df["Mes"]
    del df["Día"]

    # Reemplazar un valor por otro
    df['Tipo de Armas'] = df['Tipo de Armas'].replace('ND', np.nan)
    df.loc[df['Tortura'] > 1, 'Tortura'] = 1
    df.loc[df['Lesionados Civiles'] > 100, 'Lesionados Civiles'] = 1


    df = df[df['Presunto Responsable'] != "OTRO ¿CUÁL?"]
    df = df[df['Presunto Responsable'] != "BANDOLERISMO"]
    df = df[df['Presunto Responsable'] != "CRIMEN ORGANIZADO"]


    # Normalización de la Columna Target "Descripción Presunto Responsable"
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('EJÉRCITO NACIONAL', "FUERZAS PUBLICAS")
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('URABEÑOS/AUTODEFENSAS GAITANISTAS DE COLOMBIA/ÁGUILAS NEGRAS/CLAN ÚSUGA', "CLAN DEL GOLFO")
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('AUTODEFENSAS CAMPESINAS DEL CASANARE (BUITRAGUEÑOS)', "ALIANZA ORIENTE ACC ACMV")

    # Normalización de valores no identificados o similares
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('NO IDENTIFICADO', np.nan)
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('NO IDENTIFICADA', np.nan)
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('OTRO', np.nan)
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('NO APLICA', np.nan)
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('PRESENCIA GAO SIN ATRIBUCIÓN', np.nan)

    # Eliminación de Registros no Relevantes o Complejos de categorizar 
    df = df[df['Descripción Presunto Responsable'] != "COORDINADORA NACIONAL GUERRILLERA"]
    df = df[df['Descripción Presunto Responsable'] != "COORDINADORA GUERRILLERA SIMÓN BOLÍVAR"]
    df = df[df['Descripción Presunto Responsable'] != "MILICIAS"]
    df = df[df['Descripción Presunto Responsable'] != "AUTODEFENSAS CAMPESINAS NUEVA GENERACIÓN (ONG)"]
    df = df[df['Descripción Presunto Responsable'] != "QUINTÍN LAME"]
    df = df[df['Descripción Presunto Responsable'] != "BANDOLERISMO CONSERVADOR"]
    df = df[df['Descripción Presunto Responsable'] != "BANDOLERISMO LIBERAL"]
    df = df[df['Descripción Presunto Responsable'] != "GUERRILLA COMUNISTA"]
    df = df[df['Descripción Presunto Responsable'] != "BANDOLERISMO REVOLUCIONARIO"]
    df = df[df['Descripción Presunto Responsable'] != "VENEZOLANO"]

    # Cambio de Categoria por afinidad
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('DISIDENCIA ELN', "ELN")
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('DISIDENCIA FARC', "FARC")
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('DISIDENCIA EPL', "EPL")
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('ARMADA NACIONAL - EJÉRCITO NACIONAL', "FUERZAS PUBLICAS")
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('FARC/ELN', "ELN")
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('POLICÍA NACIONAL', "FUERZAS PUBLICAS")
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('EJÉRCITO NACIONAL - POLICÍA NACIONAL', "FUERZAS PUBLICAS")
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('FARC/EPL', "EPL")
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('BLOQUE METRO', "AUTODEFENSAS CAMPESINAS DE CÓRDOBA Y URABÁ (ACCU)")
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('FUERZA AÉREA', "FUERZAS PUBLICAS")
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('LOS PAISAS', "CLAN DEL GOLFO")
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('AUTODEFENSAS CAMPESINAS DE META Y VICHADA', "ALIANZA ORIENTE ACC ACMV")
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('AUTODEFENSAS DE HERNÁN GIRALDO', "AUTODEFENSAS UNIDAS DE COLOMBIA AUC")
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('LOS TANGUEROS', "AUTODEFENSAS UNIDAS DE COLOMBIA AUC")
    df['Descripción Presunto Responsable'] = df['Descripción Presunto Responsable'].replace('DAS', "FUERZAS PUBLICAS")

    # Separación de elementos sin categoria
    nulos = df.loc[df['Descripción Presunto Responsable'].isna()]
    df = df[df['Descripción Presunto Responsable'].notna()]

    # Guardar dataSets
    nulos.to_csv('sinClasificacion.csv')
    df.to_csv("preprocesado.csv")

print(df.columns)
preprocesar_dataSet(df)