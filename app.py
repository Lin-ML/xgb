# Importing ToolKits
import re
from time import sleep
import pandas as pd
import numpy as np
import xgboost as xgb

import streamlit as st
from streamlit.components.v1 import html
import warnings


def run():
    st.set_page_config(
        page_title="Diabete",
        page_icon="❤",
        layout="wide"
    )

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Function To Load Our Dataset
    @st.cache_data
    def load_model(model_path):
        return pd.read_pickle(model_path)

    model = xgb.Booster()
    model.load_model("final_xgb_model.json")  # 如果模型保存为 .json 格式

    st.markdown(
        """
    <style>
         .main {
            text-align: center;
         }
         h3{
            font-size: 25px
         }   
         .st-emotion-cache-16txtl3 h1 {
         font: bold 29px arial;
         text-align: center;
         margin-bottom: 15px

         }
         div[data-testid=stSidebarContent] {
         background-color: #111;
         border-right: 4px solid #222;
         padding: 8px!important

         }

         div.block-containers{
            padding-top: 0.5rem
         }

         .st-emotion-cache-z5fcl4{
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1.1rem;
            padding-right: 2.2rem;
            overflow-x: hidden;
         }

         .st-emotion-cache-16txtl3{
            padding: 2.7rem 0.6rem
         }

         .plot-container.plotly{
            border: 1px solid #333;
            border-radius: 6px;
         }

         div.st-emotion-cache-1r6slb0 span.st-emotion-cache-10trblm{
            font: bold 24px tahoma
         }
         div[data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }

        div[data-baseweb=select]>div{
            cursor: pointer;
            background-color: #fff;
            border: 1px solid #555;
            color: #000;
        }
        div[data-baseweb=select]>div:hover{
            border-color: #B72F39

        }

        div[data-baseweb=base-input]{
            background-color: #fff;
            border: 4px solid #444;
            border-radius: 5px;
            padding: 5px
            color: #000;
        }

        div[data-testid=stFormSubmitButton]> button{
            width: 40%;
            background-color: #111;
            border: 2px solid #B72F39;
            padding: 18px;
            border-radius: 30px;
            opacity: 0.8;
        }
        div[data-testid=stFormSubmitButton]  p{
            font-weight: bold;
            font-size : 20px
        }

        div[data-testid=stFormSubmitButton]> button:hover{
            opacity: 1;
            border: 2px solid #B72F39;
            color: #fff
        }
        .stAppViewBlockContainer{
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        }
        .st-emotion-cache-1v0mbdj {
        display: block;
        }
        .st-emotion-cache-gi0tri{
            display:none !important;
        }


    </style>
    """,
        unsafe_allow_html=True
    )

    header = st.container()
    content = st.container()

    st.write("")

    with header:
        st.title("BG Prediction 💔")
        st.write("")

    with content:
        col1 = st.columns([7, 5,3])
        
        with st.form("Predict"):
            c1, c2, c3 = col1
                
            with c1:
                age = st.number_input('Age', min_value=1, max_value=18, value=10, step=1)
                gender = st.selectbox('Gender', options=["Male", "Female"], index=0)
                max_BG = st.number_input('Maximum Blood Glucose', min_value=0.0, max_value=30.0, value=0.0, step=0.1)

            with c2:
                min_BG = st.number_input('Minimum Blood Glucose', min_value=0.0, max_value=30.0, value=0.0, step=0.1)
                mean_BG_24h = st.number_input('Mean Blood Glucose in Previous 24h', min_value=0.0, max_value=30.0, value=0.0, step=0.1)
                Num_Hypo = st.number_input('Number of Hypoglycemia Episodes', min_value=0, max_value=100, value=0, step=1)

            with c3:
                Num_Hyper = st.number_input('Number of Hyperglycemia Episodes', min_value=0, max_value=100, value=0, step=1)
                P1_BG = st.number_input('Previous Blood Glucose Level', min_value=0.0, max_value=30.0, value=0.0, step=0.1)
                P2_BG = st.number_input('Second Previous Blood Glucose Level', min_value=0.0, max_value=30.0, value=0.0, step=0.1)

                    
            predict_button = st.form_submit_button("Predict 🚀")



                # Appending All Data
            if predict_button:
                gender_code = 0 if gender == "Male" else 1  # 根据性别
                new_data = np.array([[age, max_BG, min_BG, mean_BG_24h, Num_Hypo, Num_Hyper, P1_BG, P2_BG, gender_code]]) 
               
                feature_names = ["Age",
                                 "Maximum Blood Glucose",
                                 "Minimum Blood Glucose",
                                 "Mean Blood Glucose in Previous 24h",
                                 "Number of Hypoglycemia Episodes",
                                 "Number of Hyperglycemia Episodes",
                                 "Previous Blood Glucose Level",
                                 "Second Previous Blood Glucose Level",
                                 "Gender"]
                
       
                dtest = xgb.DMatrix(new_data)  # 转换为 DMatrix     
                with st.spinner(text='Predict The Value..'):

                    predicted_value = model.predict(dtest)[0]
                    sleep(1.2)

                    st.image("imgs/heartbeat.png", caption="", width=100)
                    st.subheader(f"Predicted Value: {predicted_value:.2f}")


run()

