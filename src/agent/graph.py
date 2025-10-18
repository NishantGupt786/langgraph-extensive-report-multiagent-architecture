from __future__ import annotations
from typing import Dict, Literal
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, MessagesState
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# -----------------------------
#   AGENT LOGIC DEFINITIONS
# -----------------------------

def create_supervisor_chain():
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a supervisor managing a team of agents:
        
1. Researcher - Gathers information and data
2. Analyst - Analyzes data and provides insights  
3. Writer - Creates reports and summaries

Based on the current state, decide which agent should work next.
If the task is complete, respond with 'DONE'.

State Info:
- Has research data: {has_research}
- Has analysis: {has_analysis}
- Has report: {has_report}

Respond with ONLY: researcher / analyst / writer / DONE.
"""),
        ("human", "{task}")
    ])
    return supervisor_prompt | llm


def supervisor_agent(state: Dict) -> Dict:
    task = state.get("current_task", "No task")
    has_research = bool(state.get("research_data", ""))
    has_analysis = bool(state.get("analysis", ""))
    has_report = bool(state.get("final_report", ""))

    chain = create_supervisor_chain()
    decision = chain.invoke({
        "task": task,
        "has_research": has_research,
        "has_analysis": has_analysis,
        "has_report": has_report
    })

    decision_text = decision.content.strip().lower()

    if "done" in decision_text or has_report:
        next_agent = "end"
        msg = "✅ Supervisor: All tasks complete!"
    elif "researcher" in decision_text or not has_research:
        next_agent = "researcher"
        msg = "📋 Supervisor: Assigning task to Researcher."
    elif "analyst" in decision_text or (has_research and not has_analysis):
        next_agent = "analyst"
        msg = "📋 Supervisor: Assigning task to Analyst."
    elif "writer" in decision_text or (has_analysis and not has_report):
        next_agent = "writer"
        msg = "📋 Supervisor: Assigning task to Writer."
    else:
        next_agent = "end"
        msg = "✅ Supervisor: Task seems complete."

    return {
        "messages": [AIMessage(content=msg)],
        "next_agent": next_agent
    }


def researcher_agent(state: Dict) -> Dict:
    task = state.get("current_task", "research topic")
    prompt = f"""As a research specialist, provide comprehensive information about: {task}

Include:
1. Key facts and background
2. Current trends or developments
3. Important statistics
4. Notable examples

Be concise but thorough."""
    response = llm.invoke([HumanMessage(content=prompt)])
    data = response.content
    msg = f"🔍 Researcher: Completed research on '{task}'."
    return {
        "messages": [AIMessage(content=msg)],
        "research_data": data,
        "next_agent": "supervisor"
    }


def analyst_agent(state: Dict) -> Dict:
    research_data = state.get("research_data", "")
    task = state.get("current_task", "")
    prompt = f"""Analyze this research data for {task}:

{research_data}

Provide:
1. Key insights
2. Strategic implications
3. Risks and opportunities
4. Recommendations"""
    response = llm.invoke([HumanMessage(content=prompt)])
    analysis = response.content
    msg = "📊 Analyst: Analysis complete."
    return {
        "messages": [AIMessage(content=msg)],
        "analysis": analysis,
        "next_agent": "supervisor"
    }


def writer_agent(state: Dict) -> Dict:
    research_data = state.get("research_data", "")
    analysis = state.get("analysis", "")
    task = state.get("current_task", "")
    prompt = f"""Create a concise, professional report for {task}.

Research:
{research_data[:1000]}

Analysis:
{analysis[:1000]}

Include:
1. Executive Summary
2. Key Findings
3. Insights
4. Recommendations
5. Conclusion"""
    response = llm.invoke([HumanMessage(content=prompt)])
    report = response.content
    final_report = f"""
📄 FINAL REPORT
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Topic: {task}
{'='*50}

{report}

{'='*50}
Report compiled by Multi-Agent AI System powered by Gemini
"""
    msg = "✍️ Writer: Report complete."
    return {
        "messages": [AIMessage(content=msg)],
        "final_report": final_report,
        "next_agent": "supervisor",
        "task_complete": True
    }


# -----------------------------
#     GRAPH DEFINITION
# -----------------------------

class SupervisorState(MessagesState):
    next_agent: str = ""
    research_data: str = ""
    analysis: str = ""
    final_report: str = ""
    task_complete: bool = False
    current_task: str = ""


def router(state: SupervisorState) -> Literal["supervisor", "researcher", "analyst", "writer", "__end__"]:
    next_agent = state.get("next_agent", "supervisor")
    if next_agent == "end" or state.get("task_complete", False):
        return END
    return next_agent if next_agent in ["supervisor", "researcher", "analyst", "writer"] else "supervisor"


workflow = StateGraph(SupervisorState)
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("writer", writer_agent)
workflow.set_entry_point("supervisor")

for node in ["supervisor", "researcher", "analyst", "writer"]:
    workflow.add_conditional_edges(
        node,
        router,
        {
            "supervisor": "supervisor",
            "researcher": "researcher",
            "analyst": "analyst",
            "writer": "writer",
            END: END
        }
    )

graph = workflow.compile(name="Research-Orchestra")
response=graph.invoke(HumanMessage(content="What are the benefits and risks of AI in healthcare?"))
print(response)