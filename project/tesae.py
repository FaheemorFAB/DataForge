import os
from dotenv import load_dotenv
from google import genai

print("Current working directory:", os.getcwd())

load_dotenv(override=True)

key = os.getenv("GEMINI_API_KEY")
print("Loaded key:", repr(key))

# Create client
client = genai.Client(api_key=key)

print("\nAvailable models:\n")

for model in client.models.list():
    print(model.name)