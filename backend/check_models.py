import os
from google import genai

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please export GEMINI_API_KEY first.")
        return
        
    try:
        client = genai.Client(api_key=api_key)
        models = client.models.list()
        print("\n--- ACTIVE GEMINI FLASH MODELS ---")
        for m in models:
            name = getattr(m, "name", str(m))
            if "flash" in name.lower():
                print(f"- {name}")
    except Exception as e:
        print(f"Error fetching models: {e}")

if __name__ == "__main__":
    main()
