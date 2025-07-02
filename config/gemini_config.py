import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

if __name__ == "__main__":
    text = "Who are you ?"
    response = model.generate_content(text)
    print(response.text.strip())