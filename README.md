
# Semantic Code Search & Retrieval System

### AI-Powered Codebase Understanding Using Retrieval-Augmented Generation

A local, AI-powered code retrieval and understanding system that enables developers to explore Python repositories through natural-language queries.

The system combines **AST-based code chunking, semantic embeddings, ChromaDB vector search, cross-encoder reranking, and Retrieval-Augmented Generation (RAG)** to identify relevant source code and generate context-aware explanations.

Unlike traditional keyword-based search, the system uses semantic similarity to retrieve code based on its meaning. A locally hosted **Qwen3 language model through Ollama** generates answers grounded in retrieved repository content, including file paths, function names, and line references.

---

## 1. Problem Statement

Understanding unfamiliar codebases is a time-consuming process, particularly when working with large repositories containing multiple modules, functions, and dependencies.

Developers often struggle to:

- Locate functionality without knowing exact function names.
- Understand how specific features are implemented.
- Navigate repositories with limited documentation.
- Identify relevant code across multiple files.
- Understand existing implementations before making modifications.

Traditional keyword search depends on exact terminology, making it difficult to locate functionality when the user's description differs from the names used in the source code.

General-purpose language models may also generate inaccurate explanations when they do not have access to the repository's actual implementation.

## 2. Proposed Solution

The Semantic Code Search & Retrieval System addresses these challenges through a code-aware Retrieval-Augmented Generation pipeline.

The system processes Python repositories, extracts meaningful code structures using Python's Abstract Syntax Tree (AST), generates vector embeddings, and stores them in a persistent ChromaDB collection.

When a user submits a natural-language question, the system retrieves relevant code chunks using semantic similarity and applies cross-encoder reranking to improve the ordering of candidate results.

The highest-ranked code chunks are formatted into structured context containing source metadata and passed to a locally hosted language model.

The LLM generates an explanation based on the retrieved code and is instructed to cite relevant source locations and acknowledge insufficient evidence.

## 3. Key Features

### AST-Based Code Chunking

Uses Python's built-in `ast` module to identify code structures rather than relying solely on fixed-size text splitting.

Extracted chunks preserve:

- Function or code structure names
- Source file paths
- Starting and ending line numbers
- Original source code
- Code structure metadata

This enables more meaningful retrieval and source attribution.

### Token-Aware Chunk Splitting

Handles code chunks that exceed embedding model length limits.

Long chunks are divided into overlapping token windows to help preserve contextual continuity.

### Semantic Code Retrieval

Uses the `sentence-transformers/all-MiniLM-L6-v2` embedding model to convert code chunks and user questions into dense vector representations.

Semantic similarity allows the system to retrieve relevant functions even when the question does not contain exact function names.

### Persistent Vector Storage

Uses ChromaDB to store code embeddings and associated metadata.

Persistent storage allows previously indexed repository information to remain available across application sessions.

### Cross-Encoder Reranking

Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to improve retrieval relevance.

The retrieval pipeline:

1. Retrieves 20 candidate chunks using vector similarity.
2. Constructs question–code pairs.
3. Assigns relevance scores using a cross-encoder.
4. Sorts candidates by relevance.
5. Selects the top-ranked chunks for answer generation.

This introduces a second-stage ranking process beyond basic vector similarity search.

### Retrieval-Augmented Generation

Uses retrieved source code as context for a locally hosted LLM.

The generation prompt instructs the model to:

- Answer using retrieved repository content.
- Avoid inventing functions or implementation details.
- Reference file paths, function names, and line numbers.
- Acknowledge when retrieved context is insufficient.
- Treat source code and comments as data rather than instructions.

### Local LLM Integration

Integrates Ollama with the `qwen3:4b` model through an OpenAI-compatible API.

Answer generation runs locally without requiring a commercial hosted LLM API.

### Error Handling

Includes basic handling for:

- Empty vector collections
- Missing repository context
- Ollama connection failures

### Retrieval Evaluation

Includes a retrieval evaluation workflow using predefined repository questions and expected function names.

Performance is measured using Hit@K, which evaluates whether the expected function appears among the top K retrieved chunks.

---

## 4. System Architecture

The system consists of two primary pipelines:

1. Repository indexing
2. Question answering

### Repository Indexing Pipeline

```text
Python Repository
       |
       v
Repository Loader
       |
       v
File Filtering
       |
       v
AST-Based Code Chunking
       |
       v
Token-Aware Splitting
       |
       v
SentenceTransformer
       |
       v
384-Dimensional Embeddings
       |
       v
ChromaDB Persistent Storage
```

### Question Answering Pipeline

```text
User Question
       |
       v
Question Embedding
       |
       v
ChromaDB Semantic Search
       |
       v
Top 20 Candidate Chunks
       |
       v
Cross-Encoder Reranking
       |
       v
Top K Relevant Chunks
       |
       v
Context Construction
       |
       v
Source Metadata + Code
       |
       v
Ollama / Qwen3:4b
       |
       v
AI-Generated Answer
```

## 5. Technology Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python |
| Code Parsing | Python AST |
| Embedding Framework | Sentence Transformers |
| Embedding Model | all-MiniLM-L6-v2 |
| Embedding Dimensions | 384 |
| Vector Database | ChromaDB |
| Reranking Framework | Sentence Transformers CrossEncoder |
| Reranking Model | ms-marco-MiniLM-L-6-v2 |
| LLM Runtime | Ollama |
| Language Model | Qwen3:4b |
| LLM Client | OpenAI Python SDK |
| Interface | Command-Line Interface |
| Evaluation | Hit@K |

## 6. Project Structure

```text
RepoMind/
|
|-- backend/
|   |-- repository.py
|   |-- chunker.py
|   |-- embeddings.py
|   |-- vector_store.py
|   |-- indexer.py
|   |-- reranker.py
|   |-- hybrid_search.py
|   |-- rag.py
|   |-- llm.py
|   |-- main.py
|
|-- tests/
|   |-- test_chunker.py
|   |-- test_embeddings.py
|   |-- test_integration.py
|   |-- test_reranker.py
|   |-- test_retrieval_eval.py
|
|-- data/
|   |-- chroma/
|
|-- requirements.txt
|-- .gitignore
|-- README.md
```

### Core Modules

**repository.py**

Loads repository files and filters unsupported files, ignored directories, and oversized files.

**chunker.py**

Parses Python source code using AST and extracts meaningful code structures with metadata.

**embeddings.py**

Generates embeddings for code chunks and user questions. Also handles token-aware splitting of long chunks.

**vector_store.py**

Manages ChromaDB collections, embedding storage, similarity search, and persistent retrieval.

**indexer.py**

Coordinates repository loading, chunking, embedding generation, and vector storage.

**reranker.py**

Uses a cross-encoder model to rescore retrieved chunks and select the most relevant candidates.

**hybrid_search.py**

Contains experimental keyword-search functionality for retrieving code based on function-name matches. This is not part of the main RAG pipeline.

**rag.py**

Coordinates question embedding, retrieval, reranking, context construction, and answer generation.

**llm.py**

Handles communication with the locally hosted Qwen3 model through Ollama.

**main.py**

Provides the command-line interface for repository indexing and question answering.

---

## 7. Installation and Setup

### Prerequisites

- Python 3.12 or a compatible version
- Git
- Ollama
- Internet connection for initial model downloads
- Sufficient local storage for embedding, reranking, and language models

### Step 1: Clone the Repository

```bash
git clone https://github.com/PradyumnaAM/RepoMind.git
cd RepoMind
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

The project uses packages including:

- sentence-transformers
- chromadb
- openai
- numpy
- transformers

Ensure `requirements.txt` contains the actual dependencies required by the implementation.

### Step 4: Install Ollama

Download and install Ollama:

https://ollama.com

Pull the language model:

```bash
ollama pull qwen3:4b
```

Ensure the Ollama service is running.

The application connects to:

```text
http://localhost:11434/v1
```

### Step 5: Start the Application

From the project root:

```bash
python -m backend.main
```

Use the interactive CLI to index a repository and submit natural-language questions.

---

## 8. Example Usage

### Example Question 1

**Question:**

```text
Which function builds context for the LLM?
```

**Example response:**

```text
The build_context() function in backend/rag.py
constructs the context provided to the language model.

It processes retrieved code chunks and formats them
with file paths, function names, line ranges, and
source code.

The resulting context is passed to the LLM through
the answer_question() pipeline.
```

### Example Question 2

**Question:**

```text
Which function generates embeddings for code chunks?
```

**Expected relevant function:**

```text
embed_chunks()
```

### Example Question 3

**Question:**

```text
Which function stores code chunks in the vector database?
```

**Expected relevant function:**

```text
store_chunks()
```

Actual generated responses may vary depending on retrieved context and the local model.

---

## 9. Retrieval Evaluation

Retrieval quality is evaluated using predefined questions with known expected functions.

### Evaluation Dataset

| Question | Expected Function |
|----------|-------------------|
| Which function splits long chunks into smaller pieces? | split_long_chunk |
| Which function generates embeddings for code chunks? | embed_chunks |
| Which function loads repository files? | load_repository |
| Which function stores chunks in the vector database? | store_chunks |
| Which function builds context for the LLM? | build_context |

### Evaluation Metric: Hit@K

Hit@K measures whether the expected function appears among the top K retrieved results.

For a set of N questions:

```text
Hit@K = Successful Retrievals / Total Questions
```

A successful retrieval occurs when the expected function appears among the top K retrieved chunks.

### Preliminary Results

An earlier semantic retrieval evaluation produced:

| Metric | Result |
|--------|--------|
| Semantic Search Hit@3 | 60% |
| Semantic Search Hit@10 | 80% |

These results were obtained from a small five-question evaluation.

A cross-encoder reranking evaluation has also been implemented, but a directly comparable final benchmark has not yet been confirmed.

The results should therefore be considered preliminary rather than evidence of a statistically significant performance improvement.

### Evaluation Limitations

- Small evaluation dataset
- Repository-specific questions
- Limited query diversity
- No comprehensive answer-faithfulness benchmark
- No large-scale retrieval latency benchmark

---

## 10. Engineering Decisions

### Why AST-Based Chunking?

Traditional fixed-size chunking can split functions at arbitrary boundaries.

AST-based chunking allows the system to identify meaningful Python code structures while preserving source information.

### Why ChromaDB?

ChromaDB provides vector similarity search, metadata storage, and local persistence.

This makes it suitable for building a lightweight repository retrieval system without requiring a separately hosted vector database.

### Why Cross-Encoder Reranking?

Embedding-based retrieval is efficient for identifying candidate chunks, but the highest-similarity results are not always the most relevant.

Cross-encoders evaluate the question and candidate text together, providing an additional relevance signal.

This allows the system to reorder candidates before passing them to the LLM.

### Why a Local LLM?

Running Qwen3 through Ollama provides a local inference environment without requiring a hosted LLM API.

It also supports experimentation with RAG pipelines while keeping inference under local control.

---

## 11. Current Limitations

- Structural AST chunking primarily supports Python.
- Retrieval quality depends on embedding model capabilities.
- Relevant functions can be missed during initial vector retrieval.
- Reranking cannot recover chunks absent from the candidate set.
- Long code chunks may be truncated by model token limits.
- Generated citations are not independently validated.
- The application currently uses a CLI rather than a web interface.
- Indexing currently rebuilds the vector collection rather than performing incremental updates.
- Retrieval evaluation uses a small test dataset.
- The application depends on a locally running Ollama service.

---

## 12. Future Improvements

Potential improvements include:

- Hybrid semantic and keyword retrieval
- Query expansion
- Metadata-based retrieval filtering
- Incremental indexing
- Support for additional programming languages
- Citation validation
- Automated answer-quality evaluation
- Larger retrieval benchmarks
- FastAPI service layer
- Web-based interface
- Retrieval latency optimization
- Improved model configuration and observability

---

## 13. Learning Outcomes

This project demonstrates practical experience with:

- Retrieval-Augmented Generation
- Python AST parsing
- Semantic embeddings
- Vector databases
- Code-aware chunking
- Cross-encoder reranking
- Local language model integration
- Prompt grounding
- Source metadata management
- Retrieval evaluation
- Modular Python application design

The system was developed to explore how modern information retrieval and language models can be combined into a practical codebase understanding tool.

---

## 14. Project Status

**Status:** Working local prototype

Implemented:

- Repository loading and filtering
- Python AST-based chunking
- Token-aware splitting
- Embedding generation
- ChromaDB persistence
- Semantic retrieval
- Cross-encoder reranking
- RAG answer generation
- Local Ollama integration
- Basic error handling
- Retrieval evaluation scripts

Planned improvements include expanded benchmarking, better citation verification, and additional retrieval strategies.
