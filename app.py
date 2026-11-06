import sys, traceback
import streamlit as st

st.title("Debug page")
st.write("Python:", sys.version)
st.write("Streamlit:", st.__version__)
for name in ["numpy", "pandas", "sklearn", "joblib"]:
    try:
        mod = __import__(name)
        st.write(name, mod.__version__)
    except Exception:
        st.error(f"import {name} failed"); st.code(traceback.format_exc())

from pathlib import Path
base = Path(__file__).parent
st.write("Files here:", sorted(p.name for p in base.iterdir()))
st.write("model folder:", sorted(p.name for p in (base / "model").glob("*")) if (base / "model").exists() else "MISSING")
try:
    import joblib, hotel_features
    pipe = joblib.load(base / "model" / "hotel_cancel_pipeline.joblib")
    st.success("Model loaded OK")
except Exception:
    st.error("Model load failed"); st.code(traceback.format_exc())
