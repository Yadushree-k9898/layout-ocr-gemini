import os
import time
import json
import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from src.prompts import full_doc_prompt, field_prompt

logging.basicConfig(level=logging.INFO)

# Load API key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY missing in .env")
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY


class GeminiExtractor:
    def __init__(self, model: str = "gemini-pro", retries: int = 3, delay: int = 2):
        self.model = model
        self.retries = retries
        self.delay = delay

        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=0,
        )
        logging.info(f"✅ Initialized GeminiExtractor with LangChain model {model}")

    def _clean_json(self, text: str) -> str:
        """Remove markdown fences and ensure raw JSON only."""
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            # remove leading json keyword if exists
            text = text.replace("json", "", 1).strip()
        return text

    def _safe_json_loads(self, text: str) -> Dict[str, Any]:
        """Try to safely parse JSON, raising if invalid."""
        cleaned = self._clean_json(text)
        return json.loads(cleaned)

    def _retry_invoke(self, prompt: str) -> Dict[str, Any]:
        """Invoke Gemini with retries and ensure JSON-only response."""
        for attempt in range(self.retries):
            try:
                full_prompt = (
                    "You are a JSON-only extractor. "
                    "Return STRICT JSON ONLY.\n\n" + prompt
                )

                response = self.llm.invoke([HumanMessage(content=full_prompt)])

                # Handle response content properly
                raw_content = (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                )

                return self._safe_json_loads(raw_content)

            except Exception as e:
                logging.warning(
                    f"⚠️ Attempt {attempt+1} failed: {e}\nRaw response:\n{locals().get('raw_content','<no content>')}"
                )
                if attempt < self.retries - 1:
                    time.sleep(self.delay)

        raise RuntimeError("❌ Failed to get valid JSON from Gemini after retries")

    def extract_full(self, layout_json: Dict[str, Any]) -> Dict[str, Any]:
        """Extract full document JSON using Gemini."""
        prompt = full_doc_prompt(layout_json)
        return self._retry_invoke(prompt)

    def extract_fields(self, layout_json: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
        """Extract specific fields from document JSON using Gemini."""
        out = {}
        for f in fields:
            try:
                prompt = field_prompt(layout_json, f)
                data = self._retry_invoke(prompt)
                out[f] = data.get(f, None)
            except Exception as e:
                logging.error(f"❌ Failed to extract field {f}: {e}")
                out[f] = None
        return out
