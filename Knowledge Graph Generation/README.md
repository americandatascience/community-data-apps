# Knowledge Graph Generation

This application demonstrates how to build and visualize interactive knowledge graphs using AI. It provides a user-friendly interface for generating and exploring knowledge graphs from natural language queries.

## Features

- Interactive web interface built with Streamlit
- AI-powered knowledge graph generation using GPT-4
- Real-time graph visualization with Pyvis
- Support for incremental graph building
- Interactive graph exploration
- Export capabilities for graph data

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Anthropic API key (optional)

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys in a `.env` file:
```
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Use the application:
   - Enter your query in the sidebar
   - Add optional context for more accurate results
   - Choose between creating a new graph or adding to an existing one
   - Generate the knowledge graph
   - Interact with the visualization (drag nodes, zoom, etc.)
   - View raw data in the expandable section

## Features

### Graph Generation
- Natural language queries for graph generation
- Support for multiple entity types (Person, Organization, Product, etc.)
- Relationship extraction and mapping
- Incremental graph building

### Visualization
- Interactive node dragging
- Zoom and pan capabilities
- Color-coded node types
- Relationship labels
- Tooltips with entity information

### Data Management
- JSON export of graph data
- Graph reset functionality
- Merge capabilities for expanding existing graphs

## Example Queries

- "Who are all the individuals affiliated with OpenAI and their relationships?"
- "What are the key products and services offered by major tech companies?"
- "Show me the relationships between famous scientists and their discoveries"

## Configuration

The application supports various configuration options:
- Model selection
- Graph generation mode
- Context addition
- Visualization settings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please:
1. Check the documentation
2. Open an issue in this repository
3. Contact American Data Science support 