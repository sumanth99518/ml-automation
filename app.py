import streamlit as st
import pandas as pd
import os
import pycaret.classification as cl
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

import pycaret.regression as re
with st.sidebar:
    st.image('./Machine-Learning-No-Background.webp',width=200)
    st.title("Auto stream ml")
    choice =st.radio('choice',['Upload','Profling','ML','Download'])
    st.info('Build your own ML model')
if os.path.exists("dataset.csv"):
    df=pd.read_csv('dataset.csv',index_col=None)
def Upload_():
    file=st.file_uploader('upload your dataset')
    if file:
        df=pd.read_csv(file,index_col=None)
        df.to_csv('dataset.csv',index=None)
        st.dataframe(df)
def Profling_():
        st.title("Exploring the data")
        st.dataframe(df)
        profile_report=ProfileReport(df)
        st_profile_report(profile_report)
def ML_():
    st.title('Machine Learning  model')
    target=st.selectbox('select target value',df.columns)
    model_selection=st.selectbox('Type of problem ',['Classification','Regression'])
    if st.button("Train"):
        if model_selection=='Classification':
            cl.setup(df,target=target  )
            setup_df=cl.pull()
            st.info('this is the ml experiment')
            st.dataframe(setup_df)
            best_model=cl.compare_models()
            compare_df=cl.pull()
            st.info("This is the ml model")
            st.dataframe(compare_df)
            best_model
            st.info(f'{best_model}')
            cl.save_model(best_model,"best_model")

        if model_selection=='Regression':
            re.setup(df,target=target  )
            setup_df=re.pull()
            st.info('this is the ml experiment')
            st.dataframe(setup_df)
            st.write("hello")
            best_model=re.compare_models()
            compare_df=re.pull()
            st.info("This is the ml model")
            st.dataframe(compare_df)
            best_model
            st.info(f'{best_model}')
            cl.save_model(best_model,"best_model")
def Download_():
    st.title('Download')
    with open("best_model.pkl",'rb')as f:
         st.download_button('Download',f,'trained.pkl')
if choice=='Upload':
    Upload_()

if choice=='Profling':
    Profling_()

if choice=='ML':
    ML_()

if choice=='Download':
    Download_()