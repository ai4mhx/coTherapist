"""
Retrieval Augmented Generation (RAG) system for coTherapist.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Retrieval Augmented Generation system for enhancing responses with relevant knowledge.
    
    This system:
    - Embeds therapeutic knowledge documents
    - Stores embeddings in a vector database (FAISS)
    - Retrieves relevant context for user queries
    - Provides context to the language model
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RAG system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config['retrieval']
        self.embedding_model_name = self.config['embedding_model']
        self.chunk_size = self.config['chunk_size']
        self.chunk_overlap = self.config['chunk_overlap']
        self.top_k = self.config['top_k']
        self.similarity_threshold = self.config['similarity_threshold']
        
        self.embedding_model = None
        self.index = None
        self.documents = []
        self.metadata = []
        
        logger.info(f"Initializing RAG system with model: {self.embedding_model_name}")
        
    def load_embedding_model(self):
        """Load the sentence embedding model."""
        logger.info("Loading embedding model...")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        logger.info("Embedding model loaded successfully")
        
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
                
        return chunks
    
    def add_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        if self.embedding_model is None:
            self.load_embedding_model()
        
        logger.info(f"Adding {len(documents)} documents to knowledge base...")
        
        # Chunk documents
        all_chunks = []
        all_metadata = []
        
        for idx, doc in enumerate(documents):
            chunks = self.chunk_text(doc)
            all_chunks.extend(chunks)
            
            # Associate metadata with each chunk
            if metadata and idx < len(metadata):
                all_metadata.extend([metadata[idx]] * len(chunks))
            else:
                all_metadata.extend([{"doc_id": idx}] * len(chunks))
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
        embeddings = self.embedding_model.encode(
            all_chunks,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create or update FAISS index
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            logger.info(f"Created FAISS index with dimension {dimension}")
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        self.documents.extend(all_chunks)
        self.metadata.extend(all_metadata)
        
        logger.info(f"Knowledge base now contains {len(self.documents)} chunks")
        
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve (uses config default if None)
            
        Returns:
            List of relevant document chunks
        """
        if self.embedding_model is None:
            self.load_embedding_model()
            
        if self.index is None or len(self.documents) == 0:
            logger.warning("No documents in knowledge base")
            return []
        
        k = top_k if top_k is not None else self.top_k
        
        # Embed query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Filter by similarity threshold
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            # Convert L2 distance to similarity (lower distance = higher similarity)
            similarity = 1 / (1 + dist)
            
            if similarity >= self.similarity_threshold and idx < len(self.documents):
                results.append(self.documents[idx])
        
        logger.info(f"Retrieved {len(results)} relevant chunks for query")
        return results
    
    def save_index(self, path: str):
        """
        Save the FAISS index and documents to disk.
        
        Args:
            path: Directory path to save the index
        """
        if self.index is None:
            logger.warning("No index to save")
            return
        
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(path, "faiss.index")
        faiss.write_index(self.index, index_path)
        
        # Save documents and metadata
        import pickle
        data_path = os.path.join(path, "documents.pkl")
        with open(data_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
        
        logger.info(f"Index saved to {path}")
        
    def load_index(self, path: str):
        """
        Load a saved FAISS index and documents from disk.
        
        Args:
            path: Directory path containing the saved index
        """
        if self.embedding_model is None:
            self.load_embedding_model()
        
        # Load FAISS index
        index_path = os.path.join(path, "faiss.index")
        if not os.path.exists(index_path):
            logger.error(f"Index not found at {index_path}")
            return
        
        self.index = faiss.read_index(index_path)
        
        # Load documents and metadata
        import pickle
        data_path = os.path.join(path, "documents.pkl")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadata = data['metadata']
        
        logger.info(f"Loaded index with {len(self.documents)} documents from {path}")
    
    def load_knowledge_base_from_files(self, directory: str):
        """
        Load knowledge base from text files in a directory.
        
        Args:
            directory: Directory containing text files
        """
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return
        
        documents = []
        metadata = []
        
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
                    metadata.append({'source': filename})
        
        if documents:
            logger.info(f"Loaded {len(documents)} documents from {directory}")
            self.add_documents(documents, metadata)
        else:
            logger.warning(f"No .txt files found in {directory}")
