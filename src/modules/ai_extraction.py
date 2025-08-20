import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def query_gemini(prompt: str, api_key: str = None) -> str:
    """
    Send a prompt to Gemini API and return the response text.
    
    Args:
        prompt (str): The input text for Gemini.
        api_key (str, optional): API key. If None, loads from environment variable GEMINI_API_KEY.
    
    Returns:
        str: Response from Gemini API.
    """
    if api_key is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("❌ GEMINI_API_KEY not found. Set it in .env file or pass manually.")

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key   # Correct header format
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise error if bad response (4xx or 5xx)
        result = response.json()

        # Extract response text safely
        return result["candidates"][0]["content"]["parts"][0]["text"]

    except requests.exceptions.RequestException as e:
        return f"❌ Request failed: {e}"
    except (KeyError, IndexError):
        return f"⚠️ Unexpected response format: {result}"


if __name__ == "__main__":
    # Example usage
    prompt = "Give me three real-world applications of AI in healthcare."
    output = query_gemini(prompt)
    print("Gemini Response:\n", output)
