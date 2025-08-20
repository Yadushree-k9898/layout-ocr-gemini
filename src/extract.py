import os
import time
import json
import logging
import requests
from dotenv import load_dotenv
from typing import Dict, Any, List
from .prompts import full_doc_prompt, field_prompt

logging.basicConfig(level=logging.INFO)

# Load .env and get API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY missing in .env")


class GeminiExtractor:
    def __init__(self, model: str = "gemini-1.5-flash", retries: int = 3, delay: int = 2):
        """
        Initialize Gemini Extractor.

        Args:
            model (str): Gemini model to use (default: gemini-1.5-flash).
            retries (int): Number of retry attempts for API calls.
            delay (int): Delay (in seconds) between retries.
        """
        self.model = model
        self.retries = retries
        self.delay = delay
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        self.headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY
        }

    def _retry_invoke(self, prompt: str) -> Dict[str, Any]:
        """Send request to Gemini API with retry + JSON parsing"""
        for attempt in range(self.retries):
            try:
                logging.info(f"Attempt {attempt+1} to invoke Gemini API (model={self.model})")

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
                    "generationConfig": {
                        "responseMimeType": "application/json"
                    }
                }

                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()

                # Gemini puts output inside candidates
                candidate = data["candidates"][0]["content"]["parts"][0].get("text", "")

                # Parse JSON safely
                return json.loads(candidate)

            except Exception as e:
                logging.warning(f"⚠️ LLM failed attempt {attempt+1}: {e}")
                if attempt < self.retries - 1:
                    time.sleep(self.delay)

        raise RuntimeError(f"❌ Failed to get valid JSON from Gemini after {self.retries} attempts")

    def extract_full(self, layout_json: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all information from the document layout"""
        try:
            prompt = full_doc_prompt(layout_json)
            return self._retry_invoke(prompt)
        except Exception as e:
            logging.error(f"❌ Failed to extract full document: {str(e)}")
            raise

    def extract_fields(self, layout_json: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
        """Extract specific fields from the document layout"""
        out = {}
        for f in fields:
            try:
                prompt = field_prompt(layout_json, f)
                data = self._retry_invoke(prompt)
                out[f] = data.get(f, None)
            except Exception as e:
                logging.error(f"❌ Failed to extract field {f}: {str(e)}")
                out[f] = None
        return out


if __name__ == "__main__":
    try:
        extractor = GeminiExtractor(model="gemini-1.5-flash")
        logging.info("✅ Successfully initialized GeminiExtractor")
    except Exception as e:
        logging.error(f"❌ Failed to initialize GeminiExtractor: {str(e)}")
