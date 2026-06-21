from __future__ import annotations

import html

import streamlit as st
import yaml

from rbp_lab.config import MODEL_FEATURES_PATH
from rbp_lab.ui import configure_page, page_header, settings_panel


configure_page("Model Feature Summary")
settings_panel()

page_header(
    "Architecture and changelog",
    "Model Feature Summary",
    "Keep the simulator’s moving parts legible, and connect performance changes to the match where each idea entered.",
)

raw = MODEL_FEATURES_PATH.read_text(encoding="utf-8")
try:
    features = yaml.safe_load(raw).get("features", [])
except Exception as exc:
    st.error(f"Could not parse model_features.yaml: {exc}")
    features = []

columns = st.columns(3)
for index, feature in enumerate(features):
    introduced = feature.get("introduced_in_match")
    tag = (
        f'<span class="feature-tag">Since {html.escape(str(introduced))}</span>'
        if introduced
        else '<span class="feature-tag">Core</span>'
    )
    with columns[index % 3]:
        st.markdown(
            f"""
            <div class="feature-card">
              {tag}
              <h3>{html.escape(str(feature.get("name", "Untitled feature")))}</h3>
              <p>{html.escape(str(feature.get("description", "")))}</p>
              <small>{html.escape(str(feature.get("status", "active")).title())}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("Edit model_features.yaml"):
    edited = st.text_area("YAML", value=raw, height=440, label_visibility="collapsed")
    if st.button("Validate and save feature summary", type="primary"):
        try:
            parsed = yaml.safe_load(edited)
            if not isinstance(parsed, dict) or not isinstance(parsed.get("features"), list):
                raise ValueError("Top-level YAML must contain a features list.")
        except Exception as exc:
            st.error(f"YAML is not valid: {exc}")
        else:
            MODEL_FEATURES_PATH.write_text(edited.rstrip() + "\n", encoding="utf-8")
            st.success("Feature summary saved.")
            st.rerun()

