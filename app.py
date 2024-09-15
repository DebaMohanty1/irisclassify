import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import load_iris


def load_model():
    with open('best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def user_input_features():
    st.sidebar.header('User Input Parameters')


    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.0, 8.0, 5.0)
    sepal_width = st.sidebar.slider('Sepal width (cm)', 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 1.0)


    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }


    features = pd.DataFrame(data, index=[0])
    return features


def main():
    st.title("Iris Flower Classification App")
    st.write("This app predicts the species of Iris flower based on sepal and petal measurements.")


    model = load_model()


    input_df = user_input_features()
    st.write("User Input Parameters", input_df)


    iris = load_iris()
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)


    st.write(f"Predicted Class: {iris.target_names[prediction][0]}")
    st.write(f"Prediction Probabilities: {prediction_proba}")


if __name__ == '__main__':
    main()
