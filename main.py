import os
import sqlite3
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

# LangGraph and LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
# Persistence Import
from langgraph.checkpoint.sqlite import SqliteSaver

# 1. Configuration & API Key Integration
# We are using your provided Groq key directly as the master credential
os.environ["GROQ_API_KEY"] = "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"
api_key = os.getenv("GROQ_API_KEY")

# 2. Setup Persistence (The Memory Database)
# This creates 'checkpoints.sqlite' which stores the agent's state history
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

# 3. Initialize the Brain for Groq
# Using llama-3.3-70b-versatile for high-reasoning orchestration
llm = ChatOpenAI(
    openai_api_key=api_key,
    model_name="llama-3.3-70b-versatile",
    openai_api_base="https://api.groq.com/openai/v1"
)

# 4. Define the Shared State (The "Backpack")
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_node: str

# 5. Define the Specialist Nodes
def supervisor(state: AgentState):
    """The Logic Gate: Directs traffic based on content."""
    last_msg = state["messages"][-1].content.lower()
    
    # Check for approval to end the loop
    if "approved" in last_msg or "looks good" in last_msg:
        return {"next_node": "end"}
    
    # Decide between Coder and Reviewer
    if "coder output:" not in last_msg:
        return {"next_node": "coder"}
    return {"next_node": "reviewer"}

def coder(state: AgentState):
    """The Specialist: Writes Python logic."""
    user_request = state["messages"][0].content
    response = llm.invoke(f"Write a clean Python function for: {user_request}")
    return {"messages": [HumanMessage(content=f"CODER OUTPUT:\n{response.content}")]}

def reviewer(state: AgentState):
    """The QA: Critiques the output for potential errors."""
    code_to_review = state["messages"][-1].content
    response = llm.invoke(f"Review this code. If it's correct, say 'APPROVED'. Otherwise, fix it: {code_to_review}")
    return {"messages": [HumanMessage(content=f"REVIEWER FEEDBACK:\n{response.content}")]}

# 6. Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor)
workflow.add_node("coder", coder)
workflow.add_node("reviewer", reviewer)

workflow.add_edge(START, "supervisor")

workflow.add_conditional_edges(
    "supervisor",
    lambda x: x["next_node"],
    {
        "coder": "coder",
        "reviewer": "reviewer",
        "end": END
    }
)

workflow.add_edge("coder", "supervisor")
workflow.add_edge("reviewer", "supervisor")

# 7. Compile the App WITH Memory
app = workflow.compile(checkpointer=memory)

# --- EXECUTION ---
if __name__ == "__main__":
    # thread_id: allows the agent to recall the conversation history from SQLite
    config = {"configurable": {"thread_id": "lab7_session_v1"}}
    
    print("--- Starting Persistent Agent Swarm (Groq-Powered) ---")
    user_input = "Create a function for the Fibonacci sequence."
    
    try:
        inputs = {"messages": [HumanMessage(content=user_input)]}
        # Streaming the output so you can see the nodes executing in real-time
        for output in app.stream(inputs, config=config):
            for key, value in output.items():
                print(f"\n[NODE]: {key}")
                if "messages" in value:
                    print(value["messages"][-1].content)
    finally:
        conn.close()