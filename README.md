# GenAI Expert Chatbot API

A FastAPI-based chatbot that uses the Google Gemini API to answer questions about Artificial Intelligence and Generative AI.

The project supports multi-turn conversations, session-based memory, conversation history management, input validation, error handling, and automatic retries for temporary API service errors.

---

## Features

- Gemini API integration using the official `google-genai` Python SDK
- FastAPI backend
- GenAI Expert system prompt
- Multi-turn conversations
- Session-based conversation history
- Conversation history limit for basic context management
- Input validation using Pydantic
- Retry mechanism for temporary Gemini API errors
- Error handling
- Clear conversation history endpoint
- Automatic interactive API documentation using Swagger UI

---

## Technologies Used

- Python
- FastAPI
- Uvicorn
- Google Gemini API
- `google-genai` SDK
- Pydantic
- python-dotenv

---

## Project Structure

```text
llm-api-chatbot/
│
├── app/
│   ├── __init__.py
│   └── main.py
│
├── .env
├── .env.example
├── .gitignore
├── chatbot.py
├── README.md
├── requirements.txt
└── test_api.py