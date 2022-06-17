import streamlit as st
from PIL import Image
from classifier import predict

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Pest Classification App")
st.write("")

file_up = st.file_uploader("Upload an image", type="jpg")

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    labels = predict(file_up)

    # print out the top 5 prediction labels with scores
    for i in labels:
        st.write("Prediction: ", i[0], "| Score: ", i[1])
