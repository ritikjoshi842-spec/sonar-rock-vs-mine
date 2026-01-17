
import streamlit as st
import pickle
import numpy as np

# Load model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("ROCK OR MINE PREDICTION")
st.write("Check out this cool app that predicts whether the hidden object is rock or mine based on sonar data.")

# Text area for 60 values
user_input = st.text_area("Enter 60 values separated by commas:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.error("Please enter input values.")
    else:
        try:
            input_data = np.array([float(x) for x in user_input.split(',')])

            if len(input_data) != 60:
                st.error("Please enter exactly 60 values.")
            else:
                input_data = input_data.reshape(1, -1)
                prediction = model.predict(input_data)

                if prediction[0] == 'R':
                    st.success("🪨 The object is a Rock.")
                else:
                    st.success("💣 The object is a Mine.")

        except ValueError:
            st.error("Please enter only numeric values separated by commas.")
