import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Now configure Gemini with API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# List available models
models = genai.list_models()
for m in models:
    if "generateContent" in m.supported_generation_methods:
        print("Supported:", m.name)
