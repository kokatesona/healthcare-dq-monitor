"""
LangChain ReAct Agent — ICU Record Repair
Design decision: ReAct over simple chain — multi-step tool calls,
full trace in LangSmith, extensible with new tools.
LLM: OpenAI GPT-4o-mini (default) or Amazon Bedrock Claude 3.
Switch with env var: LLM_BACKEND=bedrock
"""

import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

DUCKDB_PATH = os.getenv("DUCKDB_PATH", "./data/healthcare.duckdb")


def get_llm():
    if os.getenv("LLM_BACKEND") == "bedrock":
        from langchain_aws import ChatBedrock
        return ChatBedrock(
            model_id=os.getenv("BEDROCK_MODEL_ID",
                               "anthropic.claude-3-sonnet-20240229-v1:0"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
    return ChatOpenAI(
        model="gpt-4o-mini", temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


@tool
def get_flagged_record(hadm_id: str) -> str:
    """Fetch a flagged ICU admission record with its DQ failure details."""
    import duckdb
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    row = con.execute("""
        SELECT s.hadm_id, s.admission_type, s.los_hours,
               s.total_missing_vitals, s.total_chart_rows,
               s.hr_oor_count, s.spo2_oor_count, s.rr_oor_count,
               s.invalid_icd_count, s.total_diagnoses,
               s.raw_dq_score, t.trust_score
        FROM main_mart.mart_patient_summary s
        LEFT JOIN ml_trust_scores t USING (hadm_id)
        WHERE s.hadm_id = ?
    """, [hadm_id]).df()
    con.close()
    if row.empty:
        return f"No record found for hadm_id={hadm_id}"
    r = row.iloc[0]
    return (
        f"Admission {hadm_id}: type={r.admission_type}, LOS={r.los_hours:.0f}h, "
        f"trust={r.trust_score:.1f}/100, dq_score={r.raw_dq_score:.1f}/100 | "
        f"missing_vitals={r.total_missing_vitals:.0f}/{r.total_chart_rows:.0f}, "
        f"hr_oor={r.hr_oor_count:.0f}, spo2_oor={r.spo2_oor_count:.0f}, "
        f"invalid_icd={r.invalid_icd_count:.0f}/{r.total_diagnoses:.0f}"
    )


@tool
def get_vitals_sample(hadm_id: str) -> str:
    """Get a sample of vital sign readings for an admission."""
    import duckdb
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    df = con.execute("""
        SELECT charttime, vital_name, valuenum
        FROM main_staging.stg_chartevents
        WHERE hadm_id = ?
        ORDER BY charttime LIMIT 15
    """, [hadm_id]).df()
    con.close()
    if df.empty:
        return f"No vitals found for hadm_id={hadm_id}"
    return df.to_string(index=False)


@tool
def get_trust_score(hadm_id: str) -> str:
    """Get the ML trust score for an admission."""
    import duckdb
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    row = con.execute(
        "SELECT trust_score, dq_flagged FROM ml_trust_scores WHERE hadm_id = ?",
        [hadm_id]
    ).df()
    con.close()
    if row.empty:
        return f"No trust score found for {hadm_id}"
    r = row.iloc[0]
    return f"Trust score: {r.trust_score:.1f}/100 ({'FLAGGED' if r.dq_flagged else 'CLEAN'})"


PROMPT = PromptTemplate.from_template("""You are a clinical data quality expert reviewing ICU records.
Identify data quality issues and suggest minimal, clinically-grounded repairs.

Always: (1) state the specific failure type, (2) explain clinical significance,
(3) suggest the minimal corrective action.

Tools available:
{tools}

Tool names: {tool_names}

Format:
Question: {{input}}
Thought: what to check?
Action: tool_name
Action Input: input
Observation: result
... (repeat as needed)
Thought: I have enough information
Final Answer: [failure types] | [clinical significance] | [repair action]

{agent_scratchpad}""")


def build_agent() -> AgentExecutor:
    tools  = [get_flagged_record, get_vitals_sample, get_trust_score]
    agent  = create_react_agent(get_llm(), tools, PROMPT)
    return AgentExecutor(
        agent=agent, tools=tools,
        verbose=True, max_iterations=6,
        handle_parsing_errors=True,
    )


def run_agent_on_flagged(top_n: int = 5) -> list[dict]:
    import duckdb
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    flagged = con.execute(f"""
        SELECT hadm_id FROM ml_trust_scores
        WHERE dq_flagged = true
        ORDER BY trust_score ASC LIMIT {top_n}
    """).df()
    con.close()

    if flagged.empty:
        print("No flagged admissions.")
        return []

    executor = build_agent()
    results  = []
    for hadm_id in flagged["hadm_id"].tolist():
        print(f"\n{'='*50}\nReviewing: {hadm_id}")
        try:
            result = executor.invoke({
                "input": f"Review admission {hadm_id} and provide repair recommendations."
            })
            results.append({"hadm_id": hadm_id, "output": result["output"]})
        except Exception as e:
            results.append({"hadm_id": hadm_id, "output": f"Error: {e}"})
    return results


if __name__ == "__main__":
    for r in run_agent_on_flagged(top_n=3):
        print(f"\n{r['hadm_id']}: {r['output']}")
