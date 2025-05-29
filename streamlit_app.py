import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar modelo entrenado
model = joblib.load("modelo_final_dt.joblib")  # Asegúrate de que este archivo esté en el mismo directorio

# Título
title = "Predicción de Alzheimer - Dashboard POC"
st.title(title)

# Explicación
st.markdown("""
Este dashboard permite cargar un archivo CSV con características clínicas de pacientes y obtener una predicción sobre la probabilidad de Alzheimer usando un modelo entrenado con árboles de decisión.

**Instrucciones:**
1. Sube tu archivo CSV con los datos de entrada.
2. Visualiza las predicciones y análisis del modelo.
""")

# Subir archivo
file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if file:
    data = pd.read_csv(file)
    st.subheader("Vista previa de los datos cargados")
    st.dataframe(data.head())

    # Verificar si contiene todas las columnas necesarias
    try:
        predictions = model.predict(data)
        proba = model.predict_proba(data)[:, 1]

        data['Predicción'] = predictions
        data['Probabilidad Alzheimer'] = proba

        st.subheader("Resultados de la predicción")
        st.dataframe(data[['Predicción', 'Probabilidad Alzheimer']].head(10))

        st.subheader("Distribución de predicciones")
        pred_counts = data['Predicción'].value_counts().rename({0: 'No Alzheimer', 1: 'Alzheimer'})
        fig = px.pie(values=pred_counts.values, names=pred_counts.index, title="Distribución de Predicciones")
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error al predecir: {e}")

# Métricas del modelo sobre datos de entrenamiento
def mostrar_metricas():
    st.subheader("Métricas del modelo (sobre todos los datos de entrenamiento)")

    y_true = joblib.load("y_true.joblib")
    y_pred = joblib.load("y_pred.joblib")
    y_proba = joblib.load("y_proba.joblib")

    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Matriz de confusión")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Sí"], yticklabels=["No", "Sí"])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    st.pyplot(fig)

    st.subheader("Curva ROC")
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('Curva ROC')
    ax.legend()
    st.pyplot(fig)

mostrar_metricas()

st.markdown("---")
st.caption("POC desarrollado para demostrar la capacidad del modelo en predecir Alzheimer basado en datos clínicos")
