import logging
import os

import requests
import streamlit as st
from st_ner_annotate import st_ner_annotate

from utils import calculate_precision, convert_annotation_to_tuple, render_precision

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
MEDICATION_ANNOTATION = "DRUG"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("streamlit-frontend")


def get_annotations(text: str, model: str) -> list:
    try:
        response = requests.post(f"{BACKEND_URL}/predict", json={"text": text, "model": model})
        response.raise_for_status()
        return response.json().get("annotations", [])
    except requests.exceptions.RequestException as e:
        log.error(f"Request failed: {e}")
        st.error("Failed to retrieve annotations from application backend.")
        return []
    except ValueError as e:
        log.error(f"Failed to parse response: {e}")
        st.error("Received an invalid response from application backend.")
        return []


if "annotations" not in st.session_state:
    st.session_state.annotations = []  # current annotations
if "original_annotations" not in st.session_state:
    st.session_state.original_annotations = set()  # annotations originally returned by the model
if "total_annotations" not in st.session_state:
    st.session_state.total_annotations = 0  # total number of original annotations

if "prev_text" not in st.session_state:
    st.session_state.prev_text = ""  # previous text input
if "prev_model" not in st.session_state:
    st.session_state.prev_model = ""  # previous model selection

st.title("Medication Annotation App")

text = st.text_area("Enter text here:", key="text")
model = st.selectbox("Select model", ["med7", "random"], key="model")

if text != st.session_state.prev_text or model != st.session_state.prev_model:
    st.session_state.annotations = get_annotations(text, model)
    st.session_state.original_annotations = {
        convert_annotation_to_tuple(ann) for ann in st.session_state.annotations
    }
    st.session_state.total_annotations = len(st.session_state.annotations)
    st.session_state.prev_text = text
    st.session_state.prev_model = model

    log.info(f"Input text: {text}")
    log.info(
        f"Got {st.session_state.total_annotations} annotations: {st.session_state.annotations}"
    )

if text:
    annotated = st_ner_annotate(MEDICATION_ANNOTATION, text, st.session_state.annotations)

    # annotations added by the user do not affect the precision calculation
    true_positives = sum(
        1
        for ann in annotated
        if convert_annotation_to_tuple(ann) in st.session_state.original_annotations
    )
    false_positives = st.session_state.total_annotations - true_positives

    render_precision(calculate_precision(true_positives, false_positives))
