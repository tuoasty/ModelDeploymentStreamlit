import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Iris App",
    page_icon="😍",
    layout="wide"
)

@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model 

@st.cache_data
def load_data():
    df = pd.read_csv('Iris.csv')
    df = df.drop(columns=["Id"])

try:
    model = load_models() 
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    models_loaded = False

st.title("😍 Iris App")
st.write("2702255533 - Kevin Nathanael Hendarto")

st.header("Iris Species Prediction")
if models_loaded:
    col1, col2 = st.columns(2)
        
    with col1:
        petal_length = st.slider("Petal Length (1-7):", min_value=1, max_value=7, value=3)
        petal_width = st.slider("Petal Width (1-3):", min_value=1, max_value=5, value=3)
        
    with col2:
        sepal_length = st.slider("Sepal Length (4-8):", min_value=4, max_value=8, value=6)
        sepal_width = st.slider("Sepal Witdh (2-5):", min_value=2, max_value=5, value=3)

    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if st.button("Predict Species"):
        prediction = model.predict(features)[0]
            
        label = "Iris-setosa"

        if prediction == 1:
            label = "Iris-setosa"
        elif prediction == 2:
            label = "Iris-versicolor"
        else:
            label = "Iris-virginica"
        
        st.success(f"Predicted Species: {label}")
            

    st.subheader("Feature Importance in Prediction")
    feature_importance = pd.DataFrame({
        'Feature': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
    ax.set_title("Feature Importance (Random Forest)")
    st.pyplot(fig)