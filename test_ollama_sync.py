import requests
import json
import sys

def test_chat():
    url = "http://localhost:11436/api/chat"
    payload = {
        "model": "smollm2-360m",
        "messages": [
            {"role": "user", "content": "Say 'Hello from Ollama' and nothing else."}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 50
        }
    }
    
    print(f"Sending request to {url} with payload:\n{json.dumps(payload, indent=2)}")
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        
        # Try to parse as JSON to verify it's a single object
        try:
            json_response = response.json()
            print("\nSuccessfully parsed JSON response:")
            print(json.dumps(json_response, indent=2))
            
            if "message" in json_response and "content" in json_response["message"]:
                print("\n✅ SUCCESS: Received single JSON object with content.")
            else:
                print("\n❌ FAILED: JSON object missing expected 'message.content' fields.")
        except json.JSONDecodeError as e:
            print("\n❌ FAILED to parse JSON. Raw response text:")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_chat()
