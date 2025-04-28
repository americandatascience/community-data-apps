# Community Data Apps

This repository contains a collection of open-source data applications developed by the American Data Science community. These applications showcase various data science and AI capabilities that can be built using our AI IDE for Python.

## Applications

| Application | Description | Key Features | Technologies | Example |
|-------------|-------------|--------------|--------------|---------|
| **Data Enrichment with GenAI** | AI-powered dataset enrichment | • Batch processing<br>• Custom prompt templates<br>• Real-time progress tracking | Streamlit, AI21 | [Try it →](Data%20Enrichment%20with%20GenAI/) |
| **Knowledge Graph Generation** | Interactive knowledge graph builder | • Entity extraction<br>• Relationship mapping<br>• Interactive visualization | Streamlit, PyVis, OpenAI | [Try it →](Knowledge%20Graph%20Generation/) |
| **Batch Vector DB with GPU** | High-performance vector database | • GPU-accelerated embeddings<br>• Batch processing<br>• Semantic search | PyTorch, Gradio, HuggingFace | [Try it →](Batch%20Vector%20DB%20with%20GPU/) |

## Getting Started

Each application in this repository is self-contained and includes:
- A detailed README with setup instructions
- Required dependencies
- Example data (where applicable)
- Jupyter notebooks for exploration
- Python scripts for production use

### Running in American Data Science

All applications in this repository can be easily reproduced and run in the American Data Science platform:

#### Using the Dashboard

1. Visit [dashboard.amdatascience.com](https://dashboard.amdatascience.com)
2. Create a new notebook server
3. Clone this repository: `git clone https://github.com/americandatascience/community-data-apps.git`
4. Navigate to the application of your choice
5. Follow the application's README instructions

#### Using the AMDS CLI

Install the AMDS CLI:
```bash
pip install amds
```

Start a server with appropriate compute resources:
```bash
# Login to your account
amds login

# Start a server
amds server start --server-name default --environment pytorch --compute amds-medium_cpu

# Or start a server for other applications
amds servers start --server-name data-apps
```

Connect to the Alph Editor for an AI enhanced development experience:
```bash
amds alph launch
```

For more CLI commands and options, visit the [AMDS CLI documentation](https://docs.amdatascience.com/quickstart/cli). 