import os
import json
from playsound import playsound
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv

from stt import LiveSpeechToText
from tts import DoubaouTTS

# Load environment variables from .env file
load_dotenv()

# Load API Key from models.json
MODELS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.json")))

def load_system_content(filename="system_content.json"):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(file_path, 'r') as file:
        return file.read()

def chat_with_llm(prompt, history, model_name="gpt-4o-mini", system_content="system_content.json"):
    client = OpenAI(api_key=MODELS[model_name]["api_key"], base_url=MODELS[model_name]["base_url"])
    system_content = load_system_content(system_content)
    messages = [{'role': 'system', 'content': system_content}]
    for entry in history[-5:]:
        messages.append({"role": "user", "content": entry["user"]})
        messages.append({"role": "assistant", "content": entry["bot"]})
    messages.append({"role": "user", "content": prompt})
    
    try:
        completion = client.chat.completions.create(model=model_name, messages=messages)
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error during API call: {e}")
        return "Sorry, I couldn't process your request."

def load_history(filename="chat_history.json"):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(user_input, bot_response, filename="chat_history.json"):
    history = load_history(filename)
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
    model_map = {
        "1": "gpt-4o-mini",
        "2": "qwen-plus",
        "3": "deepseek-chat"
    }
    model_selection = input("Select a model (1: gpt-4o-mini, 2: qwen-plus, 3: deepseek-chat): ")
    return model_map.get(model_selection, "gpt-4o-mini")

def main():
    model_name = init_chatbot()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_chat_history_{timestamp}.json"
    history = load_history(filename)
    stt = LiveSpeechToText()

    appid = os.getenv("DOUBAO_APPID")
    access_token = os.getenv("DOUBAO_ACCESS_TOKEN")
    cluster = os.getenv("DOUBAO_CLUSTER")
    voice_type = os.getenv("VOICE_TYPE")
    tts = DoubaouTTS(appid, access_token, cluster, voice_type)

    cnt = 0
    while True:
        user_input = stt.start_recognition()
        stt.pause()
        print("User:", user_input)
        if user_input.lower() == "please quit now":
            print("Goodbye!")
            break
        response = chat_with_llm(user_input, history, model_name)
        save_path = f"audios/output_{cnt}.mp3"
        tts.synthesize(response, output_file=save_path)
        print("\nBot:", response, "\n")
        playsound(save_path)
        save_history(user_input, response, filename)
        history.append({"user": user_input, "bot": response})
        cnt += 1

if __name__ == "__main__":
    main()
