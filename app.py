# Importing ToolKits
import re
from time import sleep
import pandas as pd
import numpy as np
import xgboost as xgb

import streamlit as st
from streamlit.components.v1 import html
import warnings

column = "Maximum Blood Glucose, Minimum Blood Glucose, Mean Blood Glucose in Previous 24h, Number of Hypoglycemia Episodes, Number of Hyperglycemia Episodes, Previous Blood Glucose Level, Second Previous Blood Glucose Level, Age, Gender"
column = column.split(", ")

def run():
    st.set_page_config(
        page_title="Diabete",
        page_icon="❤",
        layout="wide"
    )
    
    st.markdown('''<style>
        .stSlider {
            padding: 1rem;
            border: 1px solid rgba(0,0,0,0.1);
            border-radius: 0.5rem;
        }
        .stSelectbox {
            padding: 1rem;
            border: 1px solid rgba(0,0,0,0.1);
            border-radius: 0.5rem;
            height: 115.375px;
        }
    </style>''', unsafe_allow_html=True)

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Function To Load Our Dataset
    @st.cache_data
    def load_model(model_path):
        return pd.read_pickle(model_path)

    model = xgb.Booster()
    model.load_model("final_xgb_model.json")  # 如果模型保存为 .json 格式



    header = st.container()
    content = st.container()

    st.write("")

    with header:
        st.markdown('<h1 style="font-size: 30px; font-weight: bold; color: black; text-align: center; border-radius: 0.5rem; background: rgba(6,127,215,0)">BG Prediction 🩸</h1>', unsafe_allow_html=True)
        st.write("---")

    with content:
        col1 = st.columns([1,1,1])
        
        with st.form("Start predict"):
            c1, c2, c3 = col1
                
            with c1:
                age = st.slider('Age', min_value=1, max_value=18, value=10, step=1)
                gender = st.selectbox('Gender', options=["Male", "Female"], index=0)
                max_BG = st.slider('Maximum Blood Glucose', min_value=0.0, max_value=30.0, value=0.0, step=0.1)

            with c2:
                min_BG = st.slider('Minimum Blood Glucose', min_value=0.0, max_value=30.0, value=0.0, step=0.1)
                mean_BG_24h = st.slider('Mean Blood Glucose in Previous 24h', min_value=0.0, max_value=30.0, value=0.0, step=0.1)
                Num_Hypo = st.slider('Number of Hypoglycemia Episodes', min_value=0, max_value=100, value=0, step=1)

            with c3:
                Num_Hyper = st.slider('Number of Hyperglycemia Episodes', min_value=0, max_value=100, value=0, step=1)
                P1_BG = st.slider('Previous Blood Glucose Level', min_value=0.0, max_value=30.0, value=0.0, step=0.1)
                P2_BG = st.slider('Second Previous Blood Glucose Level', min_value=0.0, max_value=30.0, value=0.0, step=0.1)

            c = st.columns([1,1,1])
            predict_button = c[1].form_submit_button("Predict 🚀", use_container_width=True)



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
                
                new_data = pd.DataFrame(new_data)
                new_data.columns = feature_names
                new_data = new_data[column]
                
                with st.expander("**Current input values**", True):
                    st.dataframe(new_data, hide_index=True, use_container_width=True)
                
                dtest = xgb.DMatrix(new_data)  # 转换为 DMatrix     
                with st.spinner(text='Predict The Value..'):
                    predicted_value = model.predict(dtest)[0]
                    sleep(1.2)

                    #st.image("imgs/heartbeat.png", caption="", width=100)
                    conf = (round(predicted_value-0.32*2, 2), round(predicted_value+0.32*2, 2))
                    st.markdown(f'<p style="font-size: 30px; font-weight: bold; text-align: center;">Predicted Result Value: <span style="color: red;">{predicted_value:.2f}{str(conf)}</span>mmol/L</p>', unsafe_allow_html=True)
            else:
                with st.expander("**Current input values**", True):
                    st.warning("**Not value be input to predict.**")

run()

