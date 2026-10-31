import os
from dotenv import load_dotenv
from google import genai

# Load environment variables from the .env file
load_dotenv()

# Read the API key and model name
api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")

# Check whether the required settings exist
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. Check your .env file."
    )

if not model:
    raise ValueError(
        "GEMINI_MODEL was not found. Check your .env file."
    )

# Create the Gemini API client
client = genai.Client(api_key=api_key)

try:
    # Send a test prompt to the Gemini model
    response = client.models.generate_content(
        model=model,
        contents=(
            "Explain artificial intelligence in simple English "
            "in three sentences."
        ),
    )

    # Display the AI-generated response
    print("\nGemini AI Response:\n")
    print(response.text)

except Exception as error:
    print("\nThe Gemini API request failed.")
    print("Error type:", type(error).__name__)
    print("Error details:", str(error))