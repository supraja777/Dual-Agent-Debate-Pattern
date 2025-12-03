from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

class GraphState(TypedDict):
    question: str
    pro_answer: str
    con_challenge: str
    final_answer: str
    numberOfRounds: int
    pro_confidence: float
    con_confidence: float
    final_confidence: float

class ProAgentOutput(BaseModel):
    pro_answer : str = Field(description="This field describes the pro agents judgment")
    pro_confidence: str = Field(description="This field is used by the pro agent to indicate how confident it is regarding the judgment")


class ConAgentOutput(BaseModel):
    con_answer : str = Field(description="This field describes the con agents judgment")
    con_confidence: str = Field(description="This field is used by the con agent to indicate how confident it is regarding the judgment")

class ModeratorAgentOutput(BaseModel):
    moderator_answer: str = Field(description="The Moderator's answer based on pro and con agent's replies")
    moderator_confidence: float = Field(description="This field is used by the moderator agent to indicate how confident it is regarding the judgment")

pro_prompt = PromptTemplate(
    input_variables = ["question", "last_con"],
    template = """
        You are a Pro-Agent. Provide a strong argument for: {question} 
        Opponent's last challenge : {last_con}" \
        
        Strengthen your argument and defend your position

        Provide confidence score for your answer

        pro_answer:
        pro_confidence:
    """
)


con_prompt = PromptTemplate(
    input_variables = ["pro_answer"],
    template = """
        You are a con-Agent. 

        Critically analyze and challenge the following argument: {pro_answer}
        Strengthen your argument and defend your position

        Provide confidence score for your answer

        con_answer:
        con_confidence:
    """
)

moderator_prompt = PromptTemplate(
    input_variables = ["question", "pro_answer", "con_answer", "pro_confidence", "con_confidence"],
    template = """
        Question: {question}

        Pro-Agent: {pro_answer}
        Pro-Agent Confidence: {pro_confidence}

        Con-Agent: {con_challenge}
        Con-Agent Confidence: {con_confidence}

        Provide the best final balanced answer:
        Give the final confidence score for your answer: 

        moderator_answer:
        moderator_confidence:
    """
)

def condition(state: GraphState):
    if state["numberOfRounds"] == 8:
        return "moderatorNode"
    return "proNode"

def proNode(state: GraphState):
    llm_pro_agent = pro_prompt | llm.with_structured_output(ProAgentOutput)
    input_variables = {"question": state["question"], "last_con" : state["con_challenge"]}

    response = llm_pro_agent.invoke(input_variables)
    return {"pro_answer": response.pro_answer, "pro_confidence" : response.pro_confidence}


def conNode(state: GraphState):
    llm_con_agent = con_prompt | llm.with_structured_output(ConAgentOutput)
    input_variables = {"pro_answer": state["pro_answer"]}

    response = llm_con_agent.invoke(input_variables)
    return {"numberOfRounds": state["numberOfRounds"] + 1, "con_challenge": response.con_answer, "con_confidence" : response.con_confidence}


def moderatorNode(state: GraphState):
    moderator_agent = moderator_prompt | llm.with_structured_output(ModeratorAgentOutput)
    input_variables = {
        "question" : state["question"],
        "pro_answer": state["pro_answer"], 
        "pro_confidence": state["pro_confidence"],
        "con_challenge": state["con_challenge"],
        "con_confidence": state["con_confidence"]
    }

    response = moderator_agent.invoke(input_variables)
    return {"final_answer": response.moderator_answer, "final_confidence": response.moderator_confidence}


builder = StateGraph(GraphState)

builder.add_node("proNode", proNode)
builder.add_node("conNode", conNode)
builder.add_node("moderatorNode", moderatorNode)

builder.add_edge(START, "proNode")
builder.add_edge("proNode", "conNode")
builder.add_conditional_edges("conNode", condition)
builder.add_edge("moderatorNode", END)

graph = builder.compile()

inputs = {"question": "Is AI safe for humanity?", "numberOfRounds" : 0, "con_challenge" : ""}
result = graph.invoke(inputs)

print("Final Answer: ", result["final_answer"])
print("Final Confidence: ", result["final_confidence"])
print("Number of rounds: ", result["numberOfRounds"])
