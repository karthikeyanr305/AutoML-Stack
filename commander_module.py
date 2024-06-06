import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from app import app



def commander():

    st.markdown("<h1 style='text-align: center;color: #5fb4fb;'>ML Stack</h1>", unsafe_allow_html=True)

    st.title('Identifying Fraudulent Transactions')
    st.caption('This is an interactive Web App. Feel free to explore.')
    
    app()

commander()


