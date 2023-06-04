import requests
import streamlit as st
import webbrowser
from PIL import Image

# breeds
CATEGORIES = ["Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
              "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue", "Siamese", "Sphynx"]

# initialize text
first_prediction = "First prediction"
second_prediction = "Second prediction"
third_prediction = "Third prediction"

# config of page
st.set_page_config(page_title="Cat breed classification",
                   page_icon=":cat:", layout="wide",)

# label of page
with st.container():
    st.markdown("<h2 style='text-align: center; padding-top: 0px; margin-top: 0px;'>Individual project that classifies cat breeds with machine learning</h2>",
                unsafe_allow_html=True)

# body of page
with st.container():
    # to create a line
    st.write("---")
    # to have a gap in the middle between right and left column
    left_column, middle_column, right_column = st.columns([2, 1, 3])

    # creating the left column of page
    with left_column:
        # Here write a litle bit about the site
        st.markdown(f"<h4 style='text-align: center; padding-top: 0px; margin-top: 0px;'>These models can predict 12 cat breeds</h4>",
                    unsafe_allow_html=True)
        st.markdown(f"<h4 style='text-align: center; padding-top: 0px; margin-top: 0px;'>The 12 cat breeds that can be predicted by the model are: {', '.join(CATEGORIES)}</h4>",
                    unsafe_allow_html=True)
        
        # a select box that the user can choose between 2 algorithms
        algorithm_option = st.selectbox(
            'Select the algorithm you want to use',
            ('Transfer Learning(resnet50)', 'My Model'))
        
        # more info about the project redirecting to the github of the project
        if st.button('Project info'):
            webbrowser.open_new_tab('https://github.com/projach/individual_project')

    # creating right column
    with right_column:
        holder_photo = st.empty()
        holder_button = st.empty()
        # image upload widget
        cat_photo = holder_photo.file_uploader(
            'Upload a png or a jpg photo of a cat', type=['png', 'jpg'])
        # button for prediction
        start = holder_button.button("press when you are ready")
        # if the button is pressed
        if start:
            # if the image is not empty
            if cat_photo is not None:
                # open the image
                image = Image.open(cat_photo)
                # show the image uploaded
                st.image(image, width=300)
                # try the backend call
                try:
                    files = {"file": cat_photo.getvalue()}
                    data = {"algorithm": algorithm_option}
                    res = requests.post(
                        url="http://127.0.0.1:8000/prediction?algorithm=" + algorithm_option, files=files)
                    if res.ok:
                        res_dict = res.json()
                        first_prediction = res_dict["first_prediction"]
                        second_prediction = res_dict["second_prediction"]
                        third_prediction = res_dict["third_prediction"]
                    else:
                        st.write("Some error occured")
                except ConnectionError as e:
                    st.write("Couldn't reach backend")
            else:
                st.write("You need to upload a photo of a cat to get a prediction!")

        st.markdown(f"<h3 style='text-align: left;'>First prediction: {first_prediction}</h3>",
                    unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: left;'>Second prediction: {second_prediction}</h3>",
                    unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: left;'>Third prediction: {third_prediction}</h3>",
                    unsafe_allow_html=True)


# to run enter streamlit run app.py
