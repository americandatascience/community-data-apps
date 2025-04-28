# Batch Vector DB with GPU

This application demonstrates high-performance vector database operations using GPU acceleration. It provides a semantic search interface for Wikipedia articles with efficient batch processing capabilities.

## Features

- GPU-accelerated vector embeddings
- Batch processing for improved performance
- Interactive web interface built with Gradio
- Semantic search over Wikipedia articles
- Configurable similarity thresholds
- Progress tracking and performance metrics

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU
- CUDA toolkit installed
- PyTorch with CUDA support

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python app.py [options]
```

Available options:
- `--max-docs`: Maximum number of documents to load (default: 1000)
- `--batch-size`: Batch size for processing (default: 64)
- `--model`: Model name for embeddings (default: 'sentence-transformers/all-MiniLM-L6-v2')
- `--host`: Host to run the server on (default: '0.0.0.0')
- `--port`: Port to run the server on (default: 7860)
- `--share`: Create a public link (optional)

2. Open your web browser and navigate to the provided URL (typically http://localhost:7860)

3. Use the application:
   - Enter your search query
   - Adjust the number of results
   - Set the similarity threshold
   - View the search results

## Architecture

### Components

1. **EnhancedInMemoryRetriever**
   - GPU-accelerated embedding generation
   - Efficient batch processing
   - Cosine similarity search
   - Threshold-based filtering

2. **Document Processing**
   - Wikipedia article loading
   - Metadata extraction
   - Embedding computation
   - Vector normalization

3. **Search Interface**
   - Interactive Gradio UI
   - Real-time search results
   - Configurable parameters
   - Example queries

## Performance Features

- GPU acceleration for embedding generation
- Batch processing for improved throughput
- Memory-efficient vector operations
- Progress tracking and timing metrics
- Configurable batch sizes

## Example Queries

- "What is quantum mechanics?"
- "Tell me about the solar system"
- "How does photosynthesis work?"

## Configuration

The application supports various configuration options:
- Model selection
- Batch size adjustment
- Similarity threshold
- Number of results
- Server settings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please:
1. Check the documentation
2. Open an issue in this repository
3. Contact American Data Science support 