# creating a chatbot which answers questions related to Generative AI 
import os
import time

from dotenv import load_dotenv
from google import genai

# Load settings from the .env file
load_dotenv()

# Read the API key and model name
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")

# Check whether the settings are available
if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in the .env file.")

if not model:
    raise ValueError("GEMINI_MODEL was not found in the .env file.")

# Create the Gemini API client
client = genai.Client(api_key=api_key)

# Instructions that control the chatbot's behaviour
SYSTEM_PROMPT = """
You are a Generative AI master and can explain complex topics into simpler concepts with a suitable real life example.
Answer in simple English.
Explain technical concepts step by step.
Ask helpful follow-up questions when more information is required.
If you do not know the answer, say so honestly.
"""

# Store the conversation
conversation_history = []

# Maximum number of messages to keep
MAX_HISTORY_MESSAGES = 10

# Number of retries for temporary server problems
MAX_RETRIES = 3


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

            # Check for temporary server overload
            if "503" in error_message or "UNAVAILABLE" in error_message:
                wait_time = 2 ** attempt

                print(
                    f"\nThe AI service is temporarily busy. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:
                # Raise other errors to be handled elsewhere
                raise error

    # If all retries fail
    raise RuntimeError(
        "The AI service is still unavailable after multiple attempts."
    )


def main():
    print("================================")
    print("       GenAI Expert Chatbot")
    print("================================")
    print("Type 'exit' to end the conversation.\n")

    while True:
        user_message = input("You: ").strip()

        # Exit the chatbot
        if user_message.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        # Prevent empty messages
        if not user_message:
            print("Chatbot: Please enter a message.\n")
            continue

        # Add the user's message to history
        conversation_history.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        try:
            # Build the complete conversation context
            conversation_text = SYSTEM_PROMPT + "\n\nConversation:\n"

            for message in conversation_history:
                role = message["role"]
                content = message["content"]

                conversation_text += f"{role}: {content}\n"

            # Get a response from the AI
            chatbot_response = get_ai_response(conversation_text)

            # Display the response
            print(f"\nChatbot: {chatbot_response}\n")

            # Add the chatbot's response to history
            conversation_history.append(
                {
                    "role": "assistant",
                    "content": chatbot_response
                }
            )

            # Keep only the most recent messages
            if len(conversation_history) > MAX_HISTORY_MESSAGES:
                conversation_history[:] = conversation_history[
                    -MAX_HISTORY_MESSAGES:
                ]

        except ValueError as error:
            print(f"\nConfiguration error: {error}\n")
            conversation_history.pop()

        except RuntimeError as error:
            print(f"\nService error: {error}\n")
            conversation_history.pop()

        except Exception as error:
            print("\nAn unexpected error occurred.")
            print("Error type:", type(error).__name__)
            print("Error details:", str(error))

            # Remove the user message because no response was received
            conversation_history.pop()

            print()


if __name__ == "__main__":
    main()