# Chatbot Playground

This is a simple chatbot that runs locally using the different LLM API (openai, qwen and deepseek). It supports context-aware conversations by maintaining chat history and is designed to work on both Linux and Windows.

## Installation

### Using Conda (Preferred)
```bash
conda create --name chatbot_env python=3.8 -y
conda activate chatbot_env
pip install -r requirements.txt
```

## Procedures (Important)
1. Enter api_key for each model into the example models.json.
2. If proxies are needed, enter the correct proxies into .env file.
3. Enter your system role message into the example system_content.json.

## Running the Chatbot
```bash
python main.py
```

## Notes
- Conversation will be saved to `chat_history.json`.
- Works on both Linux and Windows.
