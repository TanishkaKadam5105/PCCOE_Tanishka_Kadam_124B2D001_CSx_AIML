Model, Prompts and Configuration

Embedding Model
- Model: all-MiniLM-L6-v2
- Library: Sentence Transformers
- Purpose: Convert document passages and user questions into embeddings for semantic retrieval.

 Vector Database
- Technology: ChromaDB
- Purpose: Store and retrieve document passages, embeddings, and page metadata.

Answer Generation Model
- Provider: Google Gemini API
- Model identifier: Copy the exact identifier used in Code/app.py.

Prompt Configuration
The prompt instructs the language model to answer using retrieved document evidence and acknowledge when the available information is insufficient. Include the exact prompt from Code/app.py in this folder.

 Security
API credentials are stored in environment configuration. Never include API keys or the .env file in the submission ZIP or public repository.