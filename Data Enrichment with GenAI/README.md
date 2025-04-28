# Data Enrichment with GenAI

This application demonstrates how to enrich datasets using Generative AI. It provides a user-friendly interface for processing and enhancing text data with AI-powered features.

## Features

- Interactive web interface built with Streamlit
- Multiple pre-built prompt templates for common enrichment tasks
- Custom prompt support for specialized use cases
- Batch processing with configurable batch sizes
- Real-time progress tracking
- Export enriched data to CSV

## Prerequisites

- Python 3.8 or higher
- AI21 API key (for accessing the AI models)

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your AI21 API key:
```bash
export AI21_API_KEY='your-api-key-here'
```

## Usage

1. Run the application:
```bash
streamlit run app.py --server.port 5000
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:5000)

3. Follow the on-screen instructions:
   - Choose a prompt template or write your own
   - Upload your CSV file
   - Select columns to enrich
   - Name your output column
   - Run the enrichment process
   - Download the enriched data

## Example Data

The repository includes a sample `drug_reviews.csv` file that you can use to test the application.

## Available Prompt Templates

- Sentiment Analysis: Extracts sentiment as Positive/Negative
- Summarization: Summarizes content in one sentence
- Entity Extraction: Extracts drug names from text
- Custom: Write your own prompt for specialized enrichment

## Configuration

You can configure the following parameters in the sidebar:
- Model selection
- Batch size for processing
- Custom prompts
- Output column naming

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please:
1. Check the documentation
2. Open an issue in this repository
3. Contact American Data Science support 