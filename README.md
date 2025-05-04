# Multi-Agent Framework for Solving Logic-Intensive Grid-Puzzle Problems

This repository contains research and implementation of multi-agent frameworks designed to solve logic-intensive grid puzzles and other constraint satisfaction problems.

## Project Overview

This project explores how collaborative AI agents can solve complex puzzles through:
- Systematic reasoning and constraint propagation
- Knowledge sharing between specialized agents
- Self-refinement of solution strategies
- Hybrid approaches combining symbolic and neural methods

## Installation

### Prerequisites
- Python 3.9+
- pip

### Setup Instructions
1. Clone this repository:
   ```bash
   git clone https://github.com/anirudh97asu/Developing-Multi-Agent-Frameworks-for-Solving-Logic-Intensive-Grid-Puzzle-Problems.git
   cd multi-agent-grid-puzzles
   ```
2. Install dependencies from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your API keys:
   ```bash
   GEMINI_API_KEY=your_gemini_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   ```
   Note: You'll need access to the Gemini flash-exp model and Groq API. Sign up at the Google AI Studio and Groq platform if you don't already have access.

## Multi-Agent System Architecture

The framework implements several approaches to multi-agent problem solving:

### Self-Refinement System
Located in `notebooks/self_refinement/`, these notebooks demonstrate how a single agent can:
- Generate initial solution strategies
- Evaluate and critique its own reasoning
- Iteratively refine approaches based on feedback
- Overcome local optima through structured reflection

### Two-Agent Collaborative System
Located in `notebooks/two_agent_system/`, these notebooks showcase:
- Specialized agent roles (constraint propagator and heuristic solver)
- Structured communication protocols between Groq and Gemini models
- Knowledge sharing and consensus building
- Performance comparison against single-agent approaches

## Experiments

All experiments are conducted in Jupyter notebooks located in the `/notebooks` directory. These notebooks demonstrate:
- Agent collaboration strategies
- Puzzle representation and transformation
- Constraint propagation techniques
- Performance benchmarks across different puzzle types and difficulty levels

To run the experiment notebooks:
```bash
jupyter notebook notebooks/
```

**Note:** The `phase_1_notebooks/` directory contains experimental explorations and preliminary work. These notebooks are for research purposes only and should be ignored for practical implementation.

## Project Structure
```
├── notebooks/
│   ├── self_refinement/     # Self-refinement approach notebooks
│   ├── two_agent_system/    # Two-agent collaboration notebooks
│   └── phase_1_notebooks/   # Initial experiments (for reference only)
├── src/                     # Source code for the multi-agent framework
├── data/                    # Example puzzle datasets
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API keys)
└── README.md                # This file
```

## Results and Performance

The framework has been tested on various puzzle types including:
- Sudoku (standard and variants)
- Nonograms (Picross)
- Logic grid puzzles
- Crosswords

Detailed performance metrics and comparisons are available in the experiment notebooks.

## License
[License information will be added]

## Contact
[Contact information will be added]