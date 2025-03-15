# Multi-Agent Framework for Solving Logic-Intensive Grid-Puzzle Problems

🚧 **Work in Progress** 🚧

This repository contains research and implementation of multi-agent frameworks designed to solve logic-intensive grid puzzles and other constraint satisfaction problems.

## Project Status

This project is currently under active development. The framework architecture, agent communication protocols, and puzzle-solving strategies are being refined. Experimental results and documentation will be updated as progress continues.

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

3. Create a `.env` file in the project root with your Gemini API key:
   ```bash
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

   Note: You'll need access to the Gemini flash-exp model. Sign up at the Google AI Studio if you don't already have access.

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

## Project Structure

```
├── notebooks/         # Experiment notebooks
├── src/               # Source code for the multi-agent framework
├── data/           # Example puzzle datasets
├── requirements.txt   # Python dependencies
├── .env               # Environment variables (API keys)
└── README.md          # This file
```

## License

[License information will be added]

## Contact

[Contact information will be added]
