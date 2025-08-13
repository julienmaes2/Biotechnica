import os
from dotenv import load_dotenv
load_dotenv()
def get_finnhub_key():
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "FINNHUB_API_KEY" in st.secrets:
            return st.secrets["FINNHUB_API_KEY"]
    except Exception:
        pass
    return os.getenv("FINNHUB_API_KEY", "")
