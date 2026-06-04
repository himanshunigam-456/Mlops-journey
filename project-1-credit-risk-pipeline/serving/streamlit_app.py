"""Streamlit demo UI for the credit-risk scoring product.

The product-facing surface a bank analyst would actually use:
upload a customer CSV → see decisions in real time → download Excel.

Run locally:
    streamlit run project-1-credit-risk-pipeline/serving/streamlit_app.py

Then open http://localhost:8501.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

# Allow running from anywhere — make the project's scripts importable
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))

from batch_predict import score_dataframe  # noqa: E402

EXAMPLES = HERE / "examples"
SAMPLE_CSV = EXAMPLES / "sample_indian_customers.csv"
SCHEMA_YAML = EXAMPLES / "icici_schema_map.yaml"


# ── Streamlit config (page metadata) ───────────────────────────────────
st.set_page_config(
    page_title="BlueLeaf Bank — Loan Decision Service",
    page_icon="🏦",
    layout="wide",
)


# ── Cached resources ───────────────────────────────────────────────────


@st.cache_data
def load_schema() -> dict:
    return yaml.safe_load(SCHEMA_YAML.read_text())


@st.cache_data
def load_sample_csv() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


# ── Sidebar (branding + about) ─────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🏦 BlueLeaf Bank")
    st.caption("Loan Decision Service · v0.2.5")
    st.markdown("---")
    st.markdown(
        """
        **What this is**

        A batch loan-decisioning product. Upload a customer CSV in
        Indian-bank format, get back per-customer decisions with
        RBI-compliant reason codes.

        **Decision bands**

        - 🟢 **APPROVE** — auto-approved (probability default < 0.25)
        - 🟡 **REVIEW** — manual underwriter review (0.25-0.50)
        - 🔴 **REJECT** — auto-rejected with reason codes (> 0.50)

        **Behind the scenes**

        - XGBoost classifier — `credit-risk-classifier @ Staging`
        - MLflow Model Registry · DVC-tracked training data
        - Image: `ghcr.io/himanshunigam-456/credit-risk-api`
        """
    )
    st.markdown("---")
    st.warning(
        "Synthetic data for demonstration only. Not actual customer data, "
        "not affiliated with any real bank."
    )


# ── Main panel ─────────────────────────────────────────────────────────
st.title("Loan Decision Service")
st.markdown(
    "Upload a customer CSV in **Indian-bank format** → get loan decisions in Excel. "
    "Same model serves the [real-time API](http://localhost:8000/docs) and this batch UI."
)

col_upload, col_sample = st.columns([3, 1])
with col_upload:
    uploaded = st.file_uploader(
        "Customer applications (CSV)",
        type=["csv"],
        help=(
            "Columns: customer_id, full_name, pan, age, ... "
            "See `examples/sample_indian_customers.csv`"
        ),
    )
with col_sample:
    st.markdown("&nbsp;")
    use_sample = st.button("Use sample data (2,000 rows)", use_container_width=True)


# Pick input source
df_input: pd.DataFrame | None = None
input_label: str = ""

if uploaded is not None:
    df_input = pd.read_csv(uploaded)
    input_label = uploaded.name
elif use_sample:
    df_input = load_sample_csv()
    input_label = SAMPLE_CSV.name


# ── Run + render ───────────────────────────────────────────────────────
if df_input is not None:
    with st.spinner(
        f"Scoring {len(df_input):,} customers against credit-risk-classifier @ Staging..."
    ):
        schema = load_schema()
        decisions = score_dataframe(df_input, schema)

    st.success(f"Scored {len(decisions):,} customers from `{input_label}`")

    # Summary metrics (the recruiter eye-candy)
    counts = decisions["decision"].value_counts()
    pct = decisions["decision"].value_counts(normalize=True) * 100
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", f"{len(decisions):,}")
    m2.metric("APPROVE 🟢", f"{int(counts.get('APPROVE', 0)):,}", f"{pct.get('APPROVE', 0):.1f}%")
    m3.metric("REVIEW 🟡", f"{int(counts.get('REVIEW', 0)):,}", f"{pct.get('REVIEW', 0):.1f}%")
    m4.metric("REJECT 🔴", f"{int(counts.get('REJECT', 0)):,}", f"{pct.get('REJECT', 0):.1f}%")

    # Audit columns (lineage badge)
    st.caption(
        f"Model: `credit-risk-classifier v{decisions['model_version'].iloc[0]}` · "
        f"Schema map: `{decisions['schema_map_version'].iloc[0]}` · "
        f"Scored at: `{decisions['scored_at'].iloc[0]}`"
    )

    # Decision table — color-coded by band
    def _color_decision(val: str) -> str:
        return {
            "APPROVE": "background-color: #d4edda; color: #155724",
            "REVIEW": "background-color: #fff3cd; color: #856404",
            "REJECT": "background-color: #f8d7da; color: #721c24",
        }.get(val, "")

    st.subheader("Decisions")
    styled = decisions.style.map(_color_decision, subset=["decision"]).format(
        {"probability_default": "{:.3f}", "loan_amount_inr": "₹{:,.0f}"}
    )
    st.dataframe(styled, use_container_width=True, height=480)

    # Download as Excel
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        decisions.to_excel(writer, index=False, sheet_name="Decisions")
    buf.seek(0)
    st.download_button(
        label="⬇  Download Excel (loan_decisions.xlsx)",
        data=buf,
        file_name="loan_decisions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

else:
    st.info(
        "👆 Upload a CSV above, or click **Use sample data** "
        "to score the bundled 2,000-row synthetic dataset."
    )
    st.markdown("### Sample of the input format")
    st.dataframe(load_sample_csv().head(5), use_container_width=True)
