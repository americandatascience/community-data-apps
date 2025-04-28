import argparse
import gradio as gr
import textwrap
from datasets import load_dataset
from retriever import EnhancedInMemoryRetriever, Document

def load_wiki_documents(max_docs: int = 1000):
    """Load and prepare Wikipedia documents"""
    print("Loading Wikipedia dataset...")
    wiki_data = load_dataset("Cohere/wikipedia-22-12-simple-embeddings", split="train")
    
    documents = []
    for i, doc in enumerate(wiki_data):
        if i >= max_docs:
            break
        documents.append(Document(
            id=str(doc["id"]),
            content=doc["title"] + " " + doc["text"],
            metadata={
                "title": doc["title"],
                "url": doc["url"]
            }
        ))
    return documents

def create_retriever(batch_size: int = 64, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    """Initialize and prepare the retriever"""
    retriever = EnhancedInMemoryRetriever(
        model_name=model_name,
        batch_size=batch_size
    )
    return retriever

def search_docs(query: str, top_k: int = 3, threshold: float = 0.3, retriever=None):
    """Search function for Gradio interface"""
    if retriever is None:
        return "Error: Retriever not initialized"
        
    results = retriever.search(query, top_k=top_k, threshold=threshold)
    
    if not results:
        return "No results found matching the threshold criteria."
    
    output = []
    for doc, score in results:
        # Truncate content for display
        content = doc.content
        if len(content) > 300:
            content = textwrap.shorten(content, width=300, placeholder="...")
            
        result_text = f"""
Score: {score:.4f}
Title: {doc.metadata['title']}
URL: {doc.metadata['url']}
Preview: {content}
{'=' * 80}
"""
        output.append(result_text)
    
    return "\n".join(output)

def create_gradio_interface(retriever):
    """Create and configure the Gradio interface"""
    iface = gr.Interface(
        fn=lambda q, k, t: search_docs(q, k, t, retriever),
        inputs=[
            gr.Textbox(
                label="Query",
                placeholder="Enter your search query here...",
                lines=2
            ),
            gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Number of results"
            ),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.3,
                step=0.05,
                label="Similarity threshold"
            )
        ],
        outputs=gr.Textbox(label="Results", lines=10),
        title="Semantic Search Demo",
        description="Search through Wikipedia articles using semantic similarity",
        examples=[
            ["What is quantum mechanics?"],
            ["Tell me about the solar system"],
            ["How does photosynthesis work?"]
        ],
        article="""
        This demo uses a sentence transformer model to perform semantic search over Wikipedia articles.
        - Higher similarity thresholds (closer to 1.0) will return more relevant but fewer results
        - Lower thresholds will return more results but might be less relevant
        - The model uses cosine similarity to find the most semantically similar documents
        """
    )
    return iface

def main():
    parser = argparse.ArgumentParser(description='Semantic Search Demo')
    parser.add_argument('--max-docs', type=int, default=1000,
                      help='Maximum number of documents to load')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size for processing')
    parser.add_argument('--model', type=str,
                      default='sentence-transformers/all-MiniLM-L6-v2',
                      help='Model name to use for embeddings')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host to run the server on')
    parser.add_argument('--port', type=int, default=7860,
                      help='Port to run the server on')
    parser.add_argument('--share', action='store_true',
                      help='Create a public link')
    
    args = parser.parse_args()
    
    # Initialize components
    print(f"Loading model: {args.model}")
    retriever = create_retriever(args.batch_size, args.model)
    
    # Load documents
    documents = load_wiki_documents(args.max_docs)
    
    # Add documents to retriever
    print("\nComputing embeddings...")
    retriever.add_documents(documents)
    
    # Create and launch the interface
    iface = create_gradio_interface(retriever)
    iface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )

if __name__ == "__main__":
    main()