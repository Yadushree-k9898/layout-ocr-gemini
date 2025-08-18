from typing import Dict, Any, List
import json, re, os, time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from .prompts import full_doc_prompt, field_prompt
import logging

logging.basicConfig(level=logging.INFO)

def _json_from_text(text: str) -> Dict[str, Any]:
    match = re.search(r'\{[\s\S]*\}$', text.strip())
    if not match:
        raise ValueError("Model did not return valid JSON")
    return json.loads(match.group(0))

class GeminiExtractor:
    def __init__(self, model: str = "gemini-1.5-flash"):
        load_dotenv()
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY missing in .env")
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    def _retry_invoke(self, prompt: str, retries: int = 3, delay: int = 2):
        for attempt in range(retries):
            try:
                res = self.llm.invoke([HumanMessage(content=prompt)])
                return _json_from_text(res.content)
            except Exception as e:
                logging.warning(f"LLM failed attempt {attempt+1}: {e}")
                time.sleep(delay)
        raise RuntimeError("Failed to get valid response from LLM after retries")

    def extract_full(self, layout_json: Dict[str, Any]) -> Dict[str, Any]:
        prompt = full_doc_prompt(layout_json)
        return self._retry_invoke(prompt)

    def extract_fields(self, layout_json: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
        out = {}
        for f in fields:
            prompt = field_prompt(layout_json, f)
            data = self._retry_invoke(prompt)
            out[f] = data.get(f, None)
        return out
