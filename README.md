# How to use
1. Setup a virtual python env in this directory and set it as source according to your OS
    ```bash
    python -m venv .venv
    ```
    ```bash
    source .venv/bin/activate
    ```
2. Install Requirements
    ```bash
    pip install -r requirements.txt
    ```
3. Create a new file called ".env" in config folder and put your GEMINI API key as follows in the .env file
    "GEMINI_API_KEY = "your_api_key"

4. Run the main.py file
    ```bash
    python main.py
    ```
5. Insert your query and it will show the relevant info and what it got from processing that query.