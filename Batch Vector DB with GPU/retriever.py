from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import time

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: np.ndarray = None

class EnhancedInMemoryRetriever:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 batch_size: int = 64,
                 device: str = None):
        self.embedder = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedder.to(self.device)
        self.documents: List[Document] = []
        self.embeddings = None
        
    def add_documents(self, documents: List[Document], show_progress: bool = True):
        """Enhanced version with better memory management and GPU support"""
        start_time = time.time()
        self.documents.extend(documents)
        
        # Prepare content for embedding
        texts = [doc.content for doc in documents]
        
        # Compute embeddings in batches with progress bar
        embeddings = []
        with torch.no_grad():  # Disable gradient computation for inference
            for i in tqdm(range(0, len(texts), self.batch_size), disable=not show_progress):
                batch = texts[i:i + self.batch_size]
                batch_embeddings = self.embedder.encode(
                    batch,
                    convert_to_tensor=True,
                    device=self.device
                )
                embeddings.extend(batch_embeddings.cpu().numpy())
        
        # Convert to numpy array and normalize
        embeddings = np.array(embeddings)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        
        # Store embeddings and update main embedding matrix
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
            
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        duration = time.time() - start_time
        print(f"\nProcessed {len(documents)} documents in {duration:.2f} seconds")
        print(f"Average time per document: {(duration/len(documents))*1000:.2f} ms")
        
    def search(self, query: str, top_k: int = 3, threshold: float = None) -> List[tuple]:
        """Enhanced search with threshold filtering and better performance"""
        # Encode and normalize query
        with torch.no_grad():
            query_embedding = self.embedder.encode(
                query,
                convert_to_tensor=True,
                device=self.device
            ).cpu().numpy()
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Compute cosine similarity (faster with normalized vectors)
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Filter by threshold if specified
        if threshold is not None:
            mask = similarities >= threshold
            if not np.any(mask):
                return []
            top_indices = np.argsort(similarities[mask])[-top_k:][::-1]
            top_indices = np.where(mask)[0][top_indices]
        else:
            top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return top k documents with scores
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            score = similarities[idx]
            results.append((doc, score))
            
        return results
    
    def search_batch(self, queries: List[str], top_k: int = 3, threshold: float = None) -> List[List[tuple]]:
        """Batch search for multiple queries"""
        # Encode all queries at once
        with torch.no_grad():
            query_embeddings = self.embedder.encode(
                queries,
                convert_to_tensor=True,
                device=self.device
            ).cpu().numpy()
        
        # Normalize query embeddings
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1)[:, np.newaxis]
        
        # Compute similarities for all queries at once
        similarities = np.dot(query_embeddings, self.embeddings.T)
        
        # Process each query's results
        all_results = []
        for i, similarity in enumerate(similarities):
            if threshold is not None:
                mask = similarity >= threshold
                if not np.any(mask):
                    all_results.append([])
                    continue
                top_indices = np.argsort(similarity[mask])[-top_k:][::-1]
                top_indices = np.where(mask)[0][top_indices]
            else:
                top_indices = np.argsort(similarity)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                doc = self.documents[idx]
                score = similarity[idx]
                results.append((doc, score))
            all_results.append(results)
            
        return all_results