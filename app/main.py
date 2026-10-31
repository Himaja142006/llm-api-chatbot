import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from google import genai
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Read API configuration
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")

# Check configuration
if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in the .env file.")

if not model:
    raise ValueError("GEMINI_MODEL was not found in the .env file.")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Create FastAPI application
app = FastAPI(
    title="GenAI Expert Chatbot API",
    description="An API-powered chatbot using Gemini.",
    version="1.0.0"
)

# System instructions
SYSTEM_PROMPT = """
You are a Generative AI master and can explain complex topics into simpler concepts with a suitable real life example.
Answer in simple English.
Explain technical concepts step by step.
Ask helpful follow-up questions when more information is required.
If you do not know the answer, say so honestly.
"""

# Maximum number of messages to keep
MAX_HISTORY_MESSAGES = 10

# Maximum retries for temporary server problems
MAX_RETRIES = 3

# Store conversation histories for different sessions
conversation_histories = {}


# Define and validate the data expected from the user
class ChatRequest(BaseModel):
    session_id: str = Field(
        min_length=1,
        max_length=100,
        description="Unique identifier for the conversation session"
    )

    message: str = Field(
        min_length=1,
        max_length=5000,
        description="User's message to the chatbot"
    )


# Function to get a response from Gemini
def get_ai_response(conversation_text):
    """Send the conversation to Gemini and handle temporary errors."""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model,
                contents=conversation_text
            )

            return response.text

        except Exception as error:
            error_message = str(error)

            # Retry temporary server overload errors
            if "503" in error_message or "UNAVAILABLE" in error_message:
                wait_time = 2 ** attempt

                print(
                    f"The AI service is temporarily busy. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:
                # Raise other errors to be handled elsewhere
                raise error

    # All retry attempts failed
    raise RuntimeError(
        "The AI service is still unavailable after multiple attempts."
    )


# Health-check endpoint
@app.get("/")
def home():
    return {
        "message": "GenAI Expert Chatbot API is running!"
    }


# Chat endpoint
@app.post("/chat")
def chat(request: ChatRequest):
    # Create a new conversation history for a new session
    if request.session_id not in conversation_histories:
        conversation_histories[request.session_id] = []

    # Get this session's conversation history
    history = conversation_histories[request.session_id]

    # Add the user's message to the history
    history.append(
        {
            "role": "user",
            "content": request.message
        }
    )

    try:
        # Build the complete conversation context
        conversation_text = SYSTEM_PROMPT + "\n\nConversation:\n"

        for message in history:
            conversation_text += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )

        # Get AI response from Gemini
        chatbot_response = get_ai_response(conversation_text)

        # Add the chatbot's response to the history
        history.append(
            {
                "role": "assistant",
                "content": chatbot_response
            }
        )

        # Keep only the most recent messages
        if len(history) > MAX_HISTORY_MESSAGES:
            conversation_histories[request.session_id] = (
                history[-MAX_HISTORY_MESSAGES:]
            )

        # Return the response
        return {
            "session_id": request.session_id,
            "response": chatbot_response
        }

    except Exception as error:
        # Remove the user's message if no AI response was received
        if history and history[-1]["role"] == "user":
            history.pop()

        raise HTTPException(
            status_code=500,
            detail=f"Chatbot error: {str(error)}"
        )


# Clear conversation history for a specific session
@app.delete("/chat/{session_id}")
def clear_chat(session_id: str):
    # Check whether the session exists
    if session_id not in conversation_histories:
        raise HTTPException(
            status_code=404,
            detail="Session not found."
        )

    # Delete the conversation history
    del conversation_histories[session_id]

    return {
        "message": "Conversation history cleared successfully.",
        "session_id": session_id
    }