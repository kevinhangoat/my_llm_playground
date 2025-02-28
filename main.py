import os
import json
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

from stt import LiveSpeechToText, WhisperLiveSpeechToText
from tts import TextToSpeech


# Load environment variables from .env file
load_dotenv()

# Load API Key from .env file
MODELS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")))


def load_system_content(filename="system_content.json"):
    root_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(root_path, filename)
    with open(file_path, 'r') as file:
        system_content = file.read()
    return system_content
    
def chat_with_llm(prompt, history, model_name="gpt-4o-mini",system_content="system_content.json"):
    client = OpenAI(api_key = MODELS[model_name]["api_key"], base_url = MODELS[model_name]["base_url"])
    system_content = load_system_content(system_content)
    messages=[{'role': 'system', 'content': system_content}]
    for entry in history[-5:]:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["bot"]})
    messages.append({"role": "user", "content": prompt})
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    
    return completion.choices[0].message.content

def load_history(filename="chat_history.json"):
    """Load chat history from a JSON file."""
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(user_input, bot_response, filename="chat_history.json"):
    """Save chat history to a JSON file."""
    history = []
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            history = json.load(f)
    history.append({
        "timestamp": datetime.now().isoformat(),
        "user": user_input,
        "bot": bot_response
    })
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def init_chatbot():
    print("Chatbot is running. Type 'exit' to quit.")
    print("Make sure you enter your API keys.")
    print("Select a model:")
    print("1. gpt-4o-mini")
    print("2. qwen-plus")
    print("3. deepseek-chat")
    model_selection = input("Enter the number of the model you want to use: ")
    
    model_map = {
        "1": "gpt-4o-mini",
        "2": "qwen-plus",
        "3": "deepseek-chat"
    }
    
    model_name = model_map.get(model_selection, "gpt-4o-mini")
    print(f"Using model: {model_name}")
    return model_name

def main():
    model_name = init_chatbot()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_chat_history_{timestamp}.json"
    history = load_history(filename=filename)
    stt = LiveSpeechToText()
    tts = TextToSpeech()

    while True:
        user_input = stt.start_recognition()
        print("User:", user_input)
        if user_input.lower() == "please quit now":
            print("Goodbye!")
            break
        response = chat_with_llm(user_input, history, model_name, system_content="test.json")
        tts.speak(response)
        print();print("Bot:", response);print()
        save_history(user_input, response, filename=filename)
        history.append({"user": user_input, "bot": response})

if __name__ == "__main__":
    main()
