import streamlit as st

def show_parameter_note(name, param):
    with st.expander(f"{name} — Beta({param['alpha']}, {param['beta']})"):
        st.markdown(param["nota"])

def show_state_note(state):
    with st.expander(f"{state['nome']} — Multiplicador: {state['multiplicador']}"):
        st.markdown(state["nota"])
