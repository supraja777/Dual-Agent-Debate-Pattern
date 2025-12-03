from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

# Prompts
pro_prompt = ChatPromptTemplate.from_template(
    "You are the Pro-Agent. Provide a strong argument for: {question}"
)

con_prompt = ChatPromptTemplate.from_template(
    "You are the Con-Agent. Critically analyze and challenge the following argument: {pro_answer}"
)

moderator_prompt = ChatPromptTemplate.from_template(
"""
Question: {question}

Pro-Agent: {pro_answer}

Con-Agent: {con_challenge}

Provide the best final balanced answer:
"""
)

class GraphState(TypedDict):
    question: str
    pro_answer: str
    con_challenge: str
    final_answer: str


def proNode(state: GraphState):
    messages = pro_prompt.format_messages(question=state["question"])
    response = llm.invoke(messages)
    return {"pro_answer": response.content}


def conNode(state: GraphState):
    messages = con_prompt.format_messages(pro_answer=state["pro_answer"])
    response = llm.invoke(messages)
    return {"con_challenge": response.content}


def moderatorNode(state: GraphState):
    messages = moderator_prompt.format_messages(
        question=state["question"],
        pro_answer=state["pro_answer"],
        con_challenge=state["con_challenge"]
    )
    response = llm.invoke(messages)
    return {"final_answer": response.content}


builder = StateGraph(GraphState)

builder.add_node("proNode", proNode)
builder.add_node("conNode", conNode)
builder.add_node("moderatorNode", moderatorNode)

builder.add_edge(START, "proNode")
builder.add_edge("proNode", "conNode")
builder.add_edge("conNode", "moderatorNode")
builder.add_edge("moderatorNode", END)

graph = builder.compile()

inputs = {"question": "Is AI safe for humanity?"}
result = graph.invoke(inputs)

print("Final Answer:\n", result["final_answer"])
