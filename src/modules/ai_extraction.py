import requests
import json

def query_gemini(text, api_key):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key
    }
    data = {
        "contents": [
            {
                "parts": [
                    {"text": text}
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()



if __name__ == "__main__":
    sample_text = "Explain how AI works in a few words"
    api_key = "GEMINI_API_KEY"  
    result = query_gemini(sample_text, api_key)
    print(json.dumps(result, indent=4))
