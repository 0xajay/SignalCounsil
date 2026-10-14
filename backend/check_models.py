import os
import requests

def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Please export GROQ_API_KEY first.")
        return
        
    response = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    if response.status_code == 200:
        models = response.json().get("data", [])
        print("\n--- ACTIVE GROQ MODELS ---")
        for m in models:
            print(f"- {m['id']}")
    else:
        print(f"Error fetching models: {response.status_code} {response.text}")

if __name__ == "__main__":
    main()
