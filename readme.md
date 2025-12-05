## 🧠 Agent Personas

To simulate a realistic and meaningful debate, each agent is initialized with a **distinct persona** and objective:

| Agent         | Persona                 | Objective                                                        |
| ------------- | ----------------------- | ---------------------------------------------------------------- |
| **Pro Agent** | Optimistic Technologist | Emphasizes innovation, progress, and positive societal outcomes  |
| **Con Agent** | Skeptical Ethicist      | Highlights risks, misuse, bias, and long-term consequences       |
| **Moderator** | Neutral Synthesizer     | Evaluates both perspectives, weighs confidence, produces balance |

These personas shape the **tone, reasoning, and style** of each agent’s responses across rounds.

---

## 🗂️ Memory & Context Handling

Each agent maintains a **short-term memory buffer** containing its most recent points.

* Only the **last N arguments** are retained (to prevent prompt overload)
* Past arguments are **summarized and fed back** into subsequent rounds
* This allows the debate to **evolve rather than repeat**

This results in:

* Increasingly refined arguments
* Reduced redundancy
* Stronger logical progression
* More informed final synthesis

---

## ⚙️ State Management (GraphLogic)

The debate is controlled by a **shared graph state object** that tracks:

* `numberOfRounds`
* `proMemory`
* `conMemory`
* `lastProResponse`
* `lastConResponse`
* `question`

A conditional function determines the flow:

```
if numberOfRounds >= MAX_ROUNDS:
    → moderator
else:
    → pro
```

This ensures **automatic looping** and **clean termination** of the debate.

---

## 🧪 Configuration

You can easily tune the system using the following parameters:

| Variable       | Purpose                 | Recommended    |
| -------------- | ----------------------- | -------------- |
| `MAX_ROUNDS`   | Total debate rounds     | 8–10           |
| `MEMORY_LIMIT` | Past arguments stored   | 6–10           |
| `MODEL_NAME`   | LLM used                | `llama3.3-70b` |
| `TEMPERATURE`  | Creativity vs stability | 0.3–0.6        |

This allows quick adjustments based on **use-case and compute constraints**.

---