"""AT&T Spam Detector -- Streamlit application entry point."""

import streamlit as st

st.set_page_config(
    page_title="AT&T Spam Detector",
    page_icon=":shield:",
    layout="wide",
)

accueil = st.Page("1_accueil.py", title="Accueil", icon=":material/home:", default=True)
exploration = st.Page("2_exploration.py", title="Exploration", icon=":material/search:")
resultats = st.Page("3_resultats.py", title="Resultats", icon=":material/bar_chart:")
demo = st.Page("4_demo.py", title="Demo", icon=":material/play_arrow:")

pg = st.navigation([accueil, exploration, resultats, demo])
pg.run()
