import streamlit as st
from streamlit_option_menu import option_menu
import webbrowser
from app import *

def nav():
    #1. as slidebar menu
    with st.sidebar:
        selected = option_menu(
            menu_title= "AI Team 6",
            options = ["YoloV7", "YoloV5"],
            icons=['binoculars', 'binoculars-fill'],
            menu_icon="bullseye", default_index=0
        )
    url = 'https://yolov5.streamlit.app/'

    if selected == "YoloV7":
        yolov7()
    if selected == "YoloV5":
       webbrowser.open_new_tab(url)

   

    st.sidebar.subheader('Objects Detectable')
    st.sidebar.write("Strawberry flowers")
    st.sidebar.write("Unripe strawberry")
    st.sidebar.write("Duck")
    st.sidebar.write("Chicken")
    st.sidebar.write("Grapes")
    st.sidebar.write("Watermelon")