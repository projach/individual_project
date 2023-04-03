import requests
import streamlit as st
from  PIL import Image
import json
import io

def transform_image(image_file):
    with io.BytesIO() as image_bytes:
        image_file.save(image_bytes, format=image_file.format)
        image_bytes.seek(0)
        return image_bytes

first_prediction = "90% Ragdoll"
second_prediction = "5% Maine Coon Cat"
third_prediction = "5% Persian"
# topk = ""
st.set_page_config(page_title="Cat breed classification", page_icon=":cat:",layout="wide",)

with st.container():
    st.markdown("<h2 style='text-align: center; padding-top: 0px; margin-top: 0px;'>This is a dissertation project that classifies cat breeds with machine learning</h2>", 
    unsafe_allow_html=True)

with st.container():
    st.write("---")
    left_column, middle_column, right_column = st.columns([2,1,3])


    with left_column:
        slider_1_value = st.slider(
        'Select a range of values', 0, 100, 35)
        st.write('value:', slider_1_value)

        slider_2_value  = st.slider(
        'Select a range of values', 0, 100, 15)
        st.write('value:', slider_2_value)

        slider_3_value = st.slider(
        'Select a range of values', 0, 100, 25)
        st.write('value:', slider_3_value)

        algorithm_option = st.selectbox(
        'Select the algorithm you want to use',
        ('algorithm 1', 'Algorithm 2', 'algorithm 3'))

    

    with right_column:
        holder_photo = st.empty()
        holder_button = st.empty()
        cat_photo = holder_photo.file_uploader('Upload a png or a jpg photo of a cat', type=['png', 'jpg'])
        start = holder_button.button("press when you are ready")
        if start:
            holder_button.empty()
            holder_photo.empty()
            if cat_photo is not None:
                image = Image.open(cat_photo)
                st.image(image,width=300)
                try:
                    files = {"file": cat_photo.getvalue()}
                    res = requests.post(url="http://127.0.0.1:8000/prediction", files=files)
                    if res.ok:
                        print("Send request to backend")
                        res_dict = res.json()
                        first_prediction = res_dict["first_prediction"]
                        second_prediction = res_dict["second_prediction"]
                        third_prediction = res_dict["third_prediction"]
                    else:
                        st.write("Some error occured")
                except ConnectionError as e:
                    st.write("Couldn't reach backend") 
           
        
        # st.markdown(f"<h3 style='text-align: left;'>{topk}</h3>", 
        # unsafe_allow_html=True)    
        st.markdown(f"<h3 style='text-align: left;'>{first_prediction}</h3>", 
        unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: left;'>{second_prediction}</h3>", 
        unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: left;'>{third_prediction}</h3>", 
        unsafe_allow_html=True)
                
                
        
#to run enter streamlit run app.py


