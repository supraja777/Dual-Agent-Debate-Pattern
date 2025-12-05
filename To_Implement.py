"""
memory_debate_agent.py

Memory-augmented, persona-driven multi-round Pro/Con debate using:
- LangGraph (StateGraph)
- LangChain PromptTemplate + structured outputs via Groq LLM bindings
- Short-term per-debate memory + long-term persona profiles
- Confidence scoring per agent
- Optional JSON persistence of persona memory after debate

This file is intended to plug into your existing project (it follows patterns
from your earlier code: ChatGroq, PromptTemplate, llm.with_structured_output(...).invoke()).

No external persistence is required; JSON persistence is optional and commented.
"""

from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from pathlib import Path
import json

load_dotenv()

# ---------- Configuration ----------
MAX_ROUNDS = 8
MEMORY_CAP = 8  # keep last N memory items per agent
PERSIST_MEMORY_AFTER_DEBATE = False
MEMORY_DUMP_PATH = Path("agent_memory_dump.json")

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# ---------- Graph State ----------
class GraphState(TypedDict):
    question: str
    numberOfRounds: int

    pro_answer: str
    con_challenge: str
    final_answer: str

    pro_confidence: float
    con_confidence: float
    final_confidence: float

    # short-term/debate memory
    pro_memory: List[str]
    con_memory: List[str]
    debate_history: List[str]

    # persona (long-term static profile loaded at start)
    pro_persona: Dict[str, Any]
    con_persona: Dict[str, Any]
    moderator_persona: Dict[str, Any]


# ---------- Structured output models ----------
class ProAgentOutput(BaseModel):
    pro_answer: str = Field(..., description="Pro agent's argument text")
    pro_confidence: float = Field(..., description="Pro agent's self-reported confidence (0-1)")

class ConAgentOutput(BaseModel):
    con_answer: str = Field(..., description="Con agent's counter-argument text")
    con_confidence: float = Field(..., description="Con agent's self-reported confidence (0-1)")

class ModeratorAgentOutput(BaseModel):
    moderator_answer: str = Field(..., description="Moderator final answer")
    moderator_confidence: float = Field(..., description="Moderator self-reported confidence (0-1)")


# ---------- Persona definitions (long-term) ----------
# These are example persona profiles. Customize as needed.
PRO_PERSONA = {
    "title": "The Optimistic Technologist",
    "beliefs": [
        "AI is a transformative tool that can amplify human capabilities",
        "Progress paired with governance is the right path"
    ],
    "style": "confident, visionary, cites innovation & benefit examples",
    "goal": "Argue that AI can be net-positive with safeguards",
    "default_confidence_range": [0.7, 0.95]
}

CON_PERSONA = {
    "title": "The Skeptical Ethicist",
    "beliefs": [
        "AI can produce significant harms if oversight is weak",
        "We must prioritize safety and fairness"
    ],
    "style": "cautious, evidence-oriented, highlights risks and edge cases",
    "goal": "Expose risks and push for strong safeguards",
    "default_confidence_range": [0.6, 0.9]
}

MODERATOR_PERSONA = {
    "title": "The Policy-Focused Mediator",
    "beliefs": [
        "Balance between innovation and safety is required",
        "Policy + technical measures yield the best outcomes"
    ],
    "style": "neutral, policy-aware, synthesis-first",
    "goal": "Produce a balanced, implementable conclusion"
}


# ---------- Helper utilities ----------
def cap_memory_list(mem_list: List[str], cap: int = MEMORY_CAP) -> List[str]:
    """Keep only the last `cap` entries."""
    if not mem_list:
        return []
    return mem_list[-cap:]


def summarize_round_summary(pro_text: str, con_text: str, round_idx: int) -> str:
    """Create a short summary for debate_history."""
    return f"Round {round_idx}: Pro -> {pro_text[:180]} | Con -> {con_text[:180]}"


def persist_memory_to_json(state: GraphState, path: Path = MEMORY_DUMP_PATH):
    """Optional: persist persona memory and debate history to disk."""
    dump = {
        "pro_persona": state["pro_persona"],
        "con_persona": state["con_persona"],
        "moderator_persona": state["moderator_persona"],
        "pro_memory": state["pro_memory"],
        "con_memory": state["con_memory"],
        "debate_history": state["debate_history"]
    }
    path.write_text(json.dumps(dump, indent=2))


# ---------- Prompt templates (use persona + memory in system prompt) ----------
PRO_PROMPT = PromptTemplate(
    input_variables=["question", "last_con", "pro_memory", "persona"],
    template="""
You are {persona[title]}.

Beliefs:
{%- for b in persona[beliefs] %}
- {b}
{%- endfor %}

Style: {persona[style]}
Goal: {persona[goal]}

Question:
{question}

Opponent's last challenge:
{last_con}

Your recent key points (from this debate):
{pro_memory}

INSTRUCTIONS:
- Stay in character and persona.
- Do NOT repeat earlier points verbatim; attempt to evolve or strengthen your position.
- Provide a concise argument for the question.
- Then provide a confidence score between 0 and 1 on a new line as: CONFIDENCE: <value>

OUTPUT (strictly):
pro_answer:
pro_confidence:
"""
)

CON_PROMPT = PromptTemplate(
    input_variables=["pro_answer", "con_memory", "persona"],
    template="""
You are {persona[title]}.

Beliefs:
{%- for b in persona[beliefs] %}
- {b}
{%- endfor %}

Style: {persona[style]}
Goal: {persona[goal]}

Pro's latest argument:
{pro_answer}

Your recent key challenges (from this debate):
{con_memory}

INSTRUCTIONS:
- Stay in persona and produce a critical challenge which targets weaknesses or edge cases.
- Attempt to introduce new angles rather than restating old points.
- Provide a confidence score between 0 and 1 on a new line: CONFIDENCE: <value>

OUTPUT (strictly):
con_answer:
con_confidence:
"""
)

MODERATOR_PROMPT = PromptTemplate(
    input_variables=["question", "pro_answer", "con_answer", "pro_confidence", "con_confidence", "persona", "debate_history"],
    template="""
You are {persona[title]}.

Beliefs: {persona[beliefs]}
Style: {persona[style]}
Goal: {persona[goal']}

Question: {question}

Debate history (high-level):
{debate_history}

Current Pro argument (confidence: {pro_confidence}):
{pro_answer}

Current Con argument (confidence: {con_confidence}):
{con_answer}

INSTRUCTIONS:
- Synthesize both sides; weigh them using reported confidences.
- Resolve open contradictions, suggest safeguards or concrete actionable steps where appropriate.
- Provide a final answer and a final confidence score (0-1).

OUTPUT (strictly):
moderator_answer:
moderator_confidence:
"""
)


# ---------- Node implementations ----------
def proNode(state: GraphState):
    """
    Pro agent reads:
    - persona
    - last con challenge (if any)
    - its pro_memory (short-term)
    Then returns pro_answer + pro_confidence (and doesn't increment round here)
    """
    # Build prompt inputs
    last_con = state["con_challenge"] or "(none)"
    pro_memory_text = "\n".join(state["pro_memory"]) if state["pro_memory"] else "(none)"
    persona = state["pro_persona"]

    llm_pro_agent = PRO_PROMPT | llm.with_structured_output(ProAgentOutput)
    inputs = {
        "question": state["question"],
        "last_con": last_con,
        "pro_memory": pro_memory_text,
        "persona": persona
    }

    response = llm_pro_agent.invoke(inputs)

    # Normalize confidence to float between 0 and 1 if model returns anything odd
    try:
        conf = float(response.pro_confidence)
        conf = max(0.0, min(1.0, conf))
    except Exception:
        conf = 0.75

    # update pro_memory with a short snippet (cap to MEMORY_CAP)
    snippet = (response.pro_answer or "").strip()
    updated_pro_memory = state["pro_memory"] + [snippet]
    updated_pro_memory = cap_memory_list(updated_pro_memory)

    return {
        "pro_answer": response.pro_answer,
        "pro_confidence": conf,
        "pro_memory": updated_pro_memory
    }


def conNode(state: GraphState):
    """
    Con agent reads:
    - persona
    - pro_answer
    - its con_memory
    - increments numberOfRounds
    Then returns con_challenge + con_confidence + updated numberOfRounds
    """
    pro_answer = state["pro_answer"]
    con_memory_text = "\n".join(state["con_memory"]) if state["con_memory"] else "(none)"
    persona = state["con_persona"]

    llm_con_agent = CON_PROMPT | llm.with_structured_output(ConAgentOutput)
    inputs = {
        "pro_answer": pro_answer,
        "con_memory": con_memory_text,
        "persona": persona
    }

    response = llm_con_agent.invoke(inputs)

    try:
        conf = float(response.con_confidence)
        conf = max(0.0, min(1.0, conf))
    except Exception:
        conf = 0.7

    snippet = (response.con_answer or "").strip()
    updated_con_memory = state["con_memory"] + [snippet]
    updated_con_memory = cap_memory_list(updated_con_memory)

    # increment round counter
    next_round = state.get("numberOfRounds", 0) + 1

    # append round summary to debate_history
    round_summary = summarize_round_summary(state.get("pro_answer", ""), snippet, next_round)
    updated_history = state["debate_history"] + [round_summary]
    updated_history = cap_memory_list(updated_history, cap=64)  # keep longer history if desired

    return {
        "con_challenge": response.con_answer,
        "con_confidence": conf,
        "numberOfRounds": next_round,
        "con_memory": updated_con_memory,
        "debate_history": updated_history
    }


def moderatorNode(state: GraphState):
    """
    Moderator synthesizes final answer using both agents' latest outputs,
    their confidences, persona, and debate history.
    """
    persona = state["moderator_persona"]
    debate_history_text = "\n".join(state["debate_history"]) if state["debate_history"] else "(none)"

    llm_moderator = MODERATOR_PROMPT | llm.with_structured_output(ModeratorAgentOutput)
    inputs = {
        "question": state["question"],
        "pro_answer": state["pro_answer"],
        "con_answer": state["con_challenge"],
        "pro_confidence": state["pro_confidence"],
        "con_confidence": state["con_confidence"],
        "persona": persona,
        "debate_history": debate_history_text
    }

    response = llm_moderator.invoke(inputs)

    try:
        conf = float(response.moderator_confidence)
        conf = max(0.0, min(1.0, conf))
    except Exception:
        conf = 0.8

    # final answer and confidence
    return {
        "final_answer": response.moderator_answer,
        "final_confidence": conf
    }


# ---------- Graph wiring ----------
def condition(state: GraphState):
    """
    Continue looping Pro <-> Con until numberOfRounds >= MAX_ROUNDS,
    then move to moderator.
    """
    if state.get("numberOfRounds", 0) >= MAX_ROUNDS:
        return "moderatorNode"
    return "proNode"


builder = StateGraph(GraphState)

builder.add_node("proNode", proNode)
builder.add_node("conNode", conNode)
builder.add_node("moderatorNode", moderatorNode)

builder.add_edge(START, "proNode")
builder.add_edge("proNode", "conNode")
builder.add_conditional_edges("conNode", condition)
builder.add_edge("moderatorNode", END)

graph = builder.compile()


# ---------- Entry / Example run ----------
if __name__ == "__main__":
    # initial state with persona & empty memory
    initial_state: GraphState = {
        "question": "Is AI safe for humanity?",
        "numberOfRounds": 0,

        "pro_answer": "",
        "con_challenge": "",
        "final_answer": "",

        "pro_confidence": 0.0,
        "con_confidence": 0.0,
        "final_confidence": 0.0,

        "pro_memory": [],
        "con_memory": [],
        "debate_history": [],

        "pro_persona": PRO_PERSONA,
        "con_persona": CON_PERSONA,
        "moderator_persona": MODERATOR_PERSONA
    }

    print("Running debate. This may take multiple LLM calls depending on MAX_ROUNDS...")
    result = graph.invoke(initial_state)

    print("\n--- Debate Result ---")
    print("Final Answer:\n", result.get("final_answer"))
    print("Final Confidence:\t", result.get("final_confidence"))
    print("Number of rounds:\t", result.get("numberOfRounds"))
    print("\nPro Memory (last items):")
    print("\n".join(result.get("pro_memory", [])[:MEMORY_CAP]))
    print("\nCon Memory (last items):")
    print("\n".join(result.get("con_memory", [])[:MEMORY_CAP]))
    print("\nDebate History (summaries):")
    print("\n".join(result.get("debate_history", [])[:20]))

    # Optional: persist persona + short-term memory to disk for future debates
    if PERSIST_MEMORY_AFTER_DEBATE:
        persist_memory_to_json(result)
        print(f"\nMemory persisted to {MEMORY_DUMP_PATH.resolve()}")

