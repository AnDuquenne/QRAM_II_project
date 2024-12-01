import streamlit as st
left_co, cent_co, last_co = st.columns(3)
with cent_co:
    st.image("app/io/logo_BLAI.png")

st.title('Portfolio Optimization')

st.write('Portfolio optimization introducing deep learning prediction in the Black-Litterman model.')