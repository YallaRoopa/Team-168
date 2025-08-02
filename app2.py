# 🧠 IBM Granite API + Streamlit + FastAPI

import streamlit as st
import requests
import threading
import time
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ---------- IBM Granite Settings ----------
IBM_API_KEY = "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"  # <-- Check validity
IBM_URL = "https://us-south.ml.cloud.ibm.com"
MODEL_ID = "granite-13b-chat-v1"

# ---------- FastAPI Backend ----------
app = FastAPI()

class PrescriptionInput(BaseModel):
    text: str
    age: int

def call_ibm_granite(prompt):
    headers = {
        "Authorization": f"Bearer {IBM_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "model_id": MODEL_ID,
        "input": [{"role": "user", "content": prompt}],
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 200
        }
    }
    response = requests.post(
        f"{IBM_URL}/foundation-models/api/v1/text/generation",  # Verify this is your endpoint
        headers=headers,
        json=payload
    )
    if response.status_code == 200:
        return response.json()["results"][0]["generated_text"]
    else:
        return f"IBM Granite API error: {response.status_code} - {response.text}"

@app.post("/analyze/")
def analyze(data: PrescriptionInput):
    prompt = (
        f"Extract drug names from this prescription and recommend dosage for age {data.age}. "
        f"Also suggest alternatives if there are interactions.\n\n"
        f"Prescription:\n{data.text}"
    )
    granite_output = call_ibm_granite(prompt)
    return {"granite_analysis": granite_output}

# ---------- Streamlit UI ----------
def run_ui():
    st.set_page_config(page_title="AI Prescription Verifier")
    st.title("🧠 AI Medical Prescription Verifier (IBM Granite)")

    text = st.text_area("Enter prescription text:")
    age = st.number_input("Enter patient age:", 0, 120, 30)

    if st.button("Analyze with IBM Watson Granite"):
        with st.spinner("Calling IBM Granite model..."):
            try:
                res = requests.post("http://127.0.0.1:8000/analyze/", json={
                    "text": text,
                    "age": age
                })
                if res.status_code == 200:
                    result = res.json()
                    st.markdown("### 🧾 Granite Response")
                    st.code(result["granite_analysis"])
                else:
                    st.error(f"API Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

# ---------- Run Backend and UI ----------
def run_api():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

if __name__ == "__main__":
    threading.Thread(target=run_api, daemon=True).start()
    time.sleep(1)
    run_ui()
