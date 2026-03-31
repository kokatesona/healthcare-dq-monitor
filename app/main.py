"""Healthcare DQ Monitor — Streamlit Dashboard"""

import os, sys
import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

DUCKDB_PATH = os.getenv("DUCKDB_PATH", "./data/healthcare.duckdb")

st.set_page_config(page_title="Healthcare DQ Monitor", page_icon="🏥", layout="wide")
st.title("🏥 Healthcare Data Quality Monitor & AI Repair Engine")
st.caption("MIMIC-III ICU · PyTorch autoencoder + GBM trust scorer · LangChain ReAct agent")


@st.cache_data(ttl=300)
def load_summary():
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    df  = con.execute("""
        SELECT s.*, t.trust_score, t.dq_flagged
        FROM main_mart.mart_patient_summary s
        LEFT JOIN ml_trust_scores t USING (hadm_id)
    """).df()
    con.close()
    return df


tab1, tab2, tab3 = st.tabs(["Pipeline overview", "Flagged records", "MLflow metrics"])

with tab1:
    df = load_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total admissions",   f"{len(df):,}")
    c2.metric("Flagged",            f"{int(df['dq_flagged'].sum()):,}",
                                    f"{100*df['dq_flagged'].mean():.1f}%")
    c3.metric("Avg trust score",    f"{df['trust_score'].mean():.1f}/100")
    c4.metric("Avg missing vitals", f"{df['total_missing_vitals'].mean():.1f}")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Trust score distribution")
        fig = px.histogram(df, x="trust_score", nbins=40,
                           color_discrete_sequence=["#0F6E56"])
        fig.add_vline(x=60, line_dash="dash", line_color="red",
                      annotation_text="Flag threshold")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        st.subheader("DQ failures by type")
        counts = {
            "Missing vitals":  int((df["total_missing_vitals"] > 0).sum()),
            "HR out-of-range": int((df["hr_oor_count"] > 0).sum()),
            "SpO2 OOR":        int((df["spo2_oor_count"] > 0).sum()),
            "RR OOR":          int((df["rr_oor_count"] > 0).sum()),
            "Invalid ICD":     int((df["invalid_icd_count"] > 0).sum()),
        }
        fig2 = px.bar(x=list(counts.keys()), y=list(counts.values()),
                      color_discrete_sequence=["#534AB7"])
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Rule-based DQ score vs ML trust score")
    fig3 = px.scatter(
        df.sample(min(1000, len(df))),
        x="raw_dq_score", y="trust_score",
        color="dq_flagged",
        color_discrete_map={True: "#E24B4A", False: "#1D9E75"},
        opacity=0.6,
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab2:
    df      = load_summary()
    flagged = df[df["dq_flagged"] == True].sort_values("trust_score")
    st.subheader(f"{len(flagged):,} flagged admissions")

    filt = st.multiselect("Filter by failure type",
        ["Missing vitals","HR out-of-range","SpO2 OOR","Invalid ICD"])
    filtered = flagged.copy()
    if "Missing vitals"  in filt: filtered = filtered[filtered["total_missing_vitals"] > 0]
    if "HR out-of-range" in filt: filtered = filtered[filtered["hr_oor_count"] > 0]
    if "SpO2 OOR"        in filt: filtered = filtered[filtered["spo2_oor_count"] > 0]
    if "Invalid ICD"     in filt: filtered = filtered[filtered["invalid_icd_count"] > 0]

    st.dataframe(filtered[[
        "hadm_id","trust_score","admission_type","los_hours",
        "total_missing_vitals","hr_oor_count","invalid_icd_count","raw_dq_score"
    ]].reset_index(drop=True), use_container_width=True, height=350)

    st.divider()
    st.subheader("AI repair suggestion")
    sel = st.text_input("Enter hadm_id", placeholder="paste from table above")
    if sel and st.button("Run agent"):
        if not os.getenv("OPENAI_API_KEY") or "your_" in os.getenv("OPENAI_API_KEY",""):
            st.warning("Add your OPENAI_API_KEY to .env to enable the agent.")
        else:
            with st.spinner("Agent reasoning..."):
                try:
                    from src.agent.repair_agent import build_agent
                    result = build_agent().invoke({
                        "input": f"Review admission {sel} and provide repair recommendations."
                    })
                    st.success("Done")
                    st.markdown(f"**Suggestion:** {result['output']}")
                except Exception as e:
                    st.error(f"Agent error: {e}")

with tab3:
    st.subheader("MLflow experiment runs")
    uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    st.info(f"Tracking URI: `{uri}`")
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient()
        exp    = client.get_experiment_by_name(
            os.getenv("MLFLOW_EXPERIMENT_NAME", "healthcare-dq-monitor"))
        if exp:
            runs = client.search_runs([exp.experiment_id],
                                      order_by=["start_time DESC"], max_results=20)
            rows = [{"run": r.data.tags.get("mlflow.runName", r.info.run_id[:8]),
                     "status": r.info.status,
                     "roc_auc": r.data.metrics.get("roc_auc"),
                     "train_loss": r.data.metrics.get("train_loss"),
                     "threshold": r.data.metrics.get("anomaly_threshold_p95")}
                    for r in runs]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No experiment found — run training first.")
    except Exception as e:
        st.error(f"MLflow error: {e}")
    st.code("mlflow ui --backend-store-uri ./mlruns --port 5001")
