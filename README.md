---
title: Swield Chatbot
emoji: 🛍️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Swield-E-Commerce-Chatbot

A customer support chatbot built for SWIELD, a fictional e-commerce store. The chatbot uses sentiment analysis to detect the user's mood and RAG to retrieve relevant answers from a knowledge base, responding naturally and appropriately.

## How It Works

1. User sends a message via the chat widget
2. Sentiment analysis detects if the user is positive or negative
3. RAG searches the knowledge base for the most relevant answer
4. The response is generated with the appropriate tone
5. If the question is unrelated to the store it politely redirects

---

## Tech Stack
- Python
- Flask
- distilbert (sentiment analysis)
- flan-t5-small (response generation)
- LangChain + ChromaDB (RAG pipeline)
- sentence-transformers (embeddings)
- Docker
- Hugging Face Spaces (deployment)

---

## Project Files
All dependencies are listed in `requirements.txt`
