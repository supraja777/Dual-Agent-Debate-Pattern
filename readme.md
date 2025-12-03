# 🧠 Multi-Round AI Debate with Confidence Scoring

This project implements an **iterative, multi-round AI debate system** where two intelligent agents — **Pro** and **Con** — argue opposing sides of a question, and a **Moderator** agent produces a final, balanced answer along with a **confidence score**.

The system is built on a **state-driven graph architecture**, enabling dynamic loops and conditional flow control across multiple rounds of reasoning.

---

## 🎯 Purpose

Modern LLMs often produce answers without opposition or scrutiny. This project introduces **structured argumentation**, allowing responses to be:

* Challenged
* Strengthened
* Refined
* Quantified via confidence

This results in **more thoughtful, explainable, and reliable outcomes**.

---

## 🔄 How It Works

1. A **user question** is passed into the system.
2. The **Pro Agent** builds an argument supporting the question’s premise.
3. The **Con Agent** critiques and challenges the Pro Agent’s position.
4. The system **loops** through these agents for multiple rounds.
5. After reaching a defined limit, the **Moderator Agent**:

   * Synthesizes both perspectives
   * Produces a final balanced response
   * Assigns a final confidence score

The number of rounds is controlled by the internal graph state (`numberOfRounds`).

---

## 📊 Confidence Scoring

Each agent provides a **self-assessed confidence score**:

| Agent     | Role                                   |
| --------- | -------------------------------------- |
| Pro Agent | Supports the argument + confidence     |
| Con Agent | Challenges the argument + confidence   |
| Moderator | Produces the final answer + confidence |


---

## 🏗️ Architecture Overview

```
Question
   ↓
Pro Agent → Con Agent → (Repeated N times)
                          ↓
                   Moderator Agent
                          ↓
                 Final Answer + Confidence
```

The loop continues until the maximum number of rounds is reached (e.g., 8 or 10), after which the Moderator is invoked.

---

## ✅ Example Outputs

### Output after 2 rounds

* **Final Answer:**

  > AI can be safe for humanity when designed with robust safety protocols, transparency, and human oversight… A nuanced approach is necessary to ensure it benefits society as a whole.

* **Final Confidence:** 0.85

* **Number of Rounds:** 2

---

### Output after 8 rounds

* **Final Answer:**

  > AI can be safe for humanity if developed responsibly, with human-centered design, transparency, explainability, independent auditing bodies, AI literacy programs, and bias mitigation systems.

* **Final Confidence:** 0.935

* **Number of Rounds:** 8

---



