# Winter School 2026: Reinforcement Learning & Prompt Engineering ❄️🤖

Welcome to the official repository for the **Winter School 2026** workshop. This repository contains materials, code examples, and homework assignments focusing on teaching AI to "give a paw" through Reinforcement Learning, as well as mastering local Large Language Models (LLMs).

**Speaker:** Viktor Zozulia  
**Date:** February 2, 2026

---

## 📚 Agenda

1.  **Prompt Engineering**: Methodologies to guide Generative AI effectively.
2.  **Local Models**: Sovereignty, privacy, and running LLMs on consumer hardware.
3.  **Reinforcement Learning (RL)**: Core concepts, algorithms, and agent training.
4.  **Homework**: "Vibe coding" a game and training an RL agent to play it.

---

## 🧠 Module 1: Prompt Engineering

We explore advanced strategies to structure inputs (prompts) to control model output, specifically optimized for both large and smaller local models.

**Techniques Covered:**
* [cite_start]**Few-Shot Prompting:** Providing examples (classification, formatting) to guide the model[cite: 88].
* [cite_start]**Chain of Thought (CoT):** Encouraging step-by-step reasoning for logic and math puzzles[cite: 134].
* [cite_start]**Meta Prompting:** Role-playing and instructing the model to refine its own prompts[cite: 203].
* [cite_start]**Prompt Chaining:** Breaking complex tasks (e.g., summarize $\to$ extract $\to$ write post) into pipelines[cite: 323].
* [cite_start]**Tree of Thought (ToT):** Exploring multiple reasoning branches for planning and strategy[cite: 431].
* [cite_start]**ReAct (Reason + Act):** Combining internal reasoning with external tools (search, code execution)[cite: 501].

---

## 💻 Module 2: Local Models

Why run models locally? [cite_start]This module covers the trade-offs between APIs and Self-Hosting regarding control, privacy, and cost[cite: 606].

**Tools Included:**
* [cite_start]**[Ollama](https://ollama.com/):** CLI tool for easy model management and automation[cite: 624].
* [cite_start]**[LM Studio](https://lmstudio.ai/):** GUI for discovering and running local LLMs (GGUF format)[cite: 651].

**Code Examples:**
Check the `/local_models` folder for Python scripts demonstrating how to interact with models like `phi-2`, `tinyllama`, and `qwen` locally.

---

## 🎮 Module 3: Reinforcement Learning

Moving from "Pattern Recognition" to "Decision Making." [cite_start]We cover the transition from Supervised Learning to RL, where agents learn by maximizing future rewards[cite: 741].

**Core Concepts:**
* [cite_start]**The RL Loop:** Agent $\leftrightarrow$ Environment interactions (State, Action, Reward)[cite: 763].
* [cite_start]**Exploration vs. Exploitation:** Balancing trying new actions vs. maximizing known rewards[cite: 845].
* **Model-Free Methods:**
    * [cite_start]**Value-Based:** Q-Learning, Deep Q-Networks (DQN)[cite: 932].
    * [cite_start]**Policy-Based:** Optimizing policy directly (REINFORCE)[cite: 973].
    * [cite_start]**Actor-Critic:** Combining value and policy functions (PPO, SAC, A2C)[cite: 992].

---

## 🛠️ Homework: Vibe Coding & Agents

The practical component of this workshop puts theory into practice.

[cite_start]**The Assignment[cite: 1022]:**
1.  **Vibe Code a Game:** Use Prompt Engineering techniques with a local LLM to help you write a simple Python game (e.g., *Grid Coin Collector*).
2.  **Select an RL Method:** Choose the best algorithm for your game (e.g., Q-Learning for discrete spaces).
3.  **Train the Agent:** Implement the training loop and teach your agent to beat the game!

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/download) or [LM Studio](https://lmstudio.ai/download)
* Recommended libraries: `torch`, `gymnasium`, `pygame`, `ollama`

### Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/winter-school-2026-rl.git](https://github.com/your-username/winter-school-2026-rl.git)
cd winter-school-repo

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running Local Model Examples:

```bash
python local_models/ollama_example.py
```

### Playing the "Vibe-Coded" Game:
```bash
python games/grid_coin_collector.py
```

### Training the RL Agent:

(missing for now)

```bash
python rl_agent/train.py
```

### 📂 Repository Structure
├── presentation/       # Winter School 2026 Slides (PDF)

├── prompt_examples/    # Few-shot, CoT, and ReAct prompt templates

├── local_models/       # Python integration for Ollama & LM Studio

├── games/              # Simple Python games created with LLMs

└── rl_agent/           # RL algorithms and training scripts