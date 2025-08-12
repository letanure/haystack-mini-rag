Mini RAG Documentation
======================

A lightweight Retrieval-Augmented Generation (RAG) system for learning purposes.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

Features
--------

- **Dense Retrieval**: Uses Haystack v2 with sentence-transformers (MiniLM) for semantic search
- **OpenAI Generation**: Integrates with GPT-4o-mini for answer generation  
- **Smart Caching**: Disk-based embedding cache for 10x faster repeated runs
- **Multiple Document Types**: Load from JSONL, PDF, DOCX, TXT files, directories, and URLs
- **Comprehensive Evaluation**: Recall@K + answer quality metrics with 15 test cases
- **Professional CLI**: Color-coded output with various options

Quick Start
-----------

.. code-block:: bash

   # Install
   pip install -e .
   
   # Basic usage
   minirag ask "What is RAG?"
   
   # Load from different sources
   minirag ask "What is this about?" --source="research.pdf"
   minirag ask "Analyze this page" --source="https://example.com"

