import os
import json
import requests
import logging
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

class GeminiExtractor:
    def __init__(self, model: str = "gemini-1.5-flash", retries: int = 3, delay: int = 2):
        self.model = model
        self.retries = retries
        self.delay = delay
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def _retry_invoke(self, prompt: str) -> Dict[str, Any]:
        for attempt in range(self.retries):
            try:
                logging.info(f"Attempt {attempt+1} to invoke Gemini API with model={self.model}")

                payload = {
                    "contents": [
                        {
                            "parts": [
                                {
                                    "text": (
                                        "You are a JSON-only extractor. "
                                        "Return ONLY valid JSON without explanations. "
                                        "If extraction fails, return an empty JSON {}.\n\n"
                                        f"{prompt}"
                                    )
                                }
                            ]
                        }
                    ],
                    "generationConfig": {"responseMimeType": "application/json"},
                }

                response = requests.post(self.api_url, headers=HEADERS, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()

                candidate = data["candidates"][0]["content"]["parts"][0].get("text", "")
                return json.loads(candidate)

            except Exception as e:
                logging.warning(f"⚠️ LLM failed attempt {attempt+1}: {e}")
                if attempt < self.retries - 1:
                    time.sleep(self.delay)

        raise RuntimeError("❌ Failed to get valid JSON from Gemini after retries")
