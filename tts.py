import pyttsx3
import requests
import json
import uuid
import base64
from dotenv import load_dotenv
import os
import pdb

class TextToSpeech:
    def __init__(self):
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.voices = self.tts_engine.getProperty('voices')
        self.tts_engine.setProperty('voice', self.voices[7].id) 
        self.tts_engine.setProperty('volume', 1)  # Volume (0.0 to 1.0)

    def speak(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

class DoubaouTTS:
    def __init__(self, appid, access_token, cluster, voice_type):
        self.appid = appid
        self.access_token = access_token
        self.cluster = cluster
        self.voice_type = voice_type
        self.host = "openspeech.bytedance.com"
        self.api_url = f"https://{self.host}/api/v1/tts"
        self.header = {"Authorization": f"Bearer;{self.access_token}"}

    def synthesize(self, text, output_file="audios/output.mp3"):
        request_json = {
            "app": {
                "appid": self.appid,
                "token": "access_token",
                "cluster": self.cluster
            },
            "user": {
                "uid": "388808087185088"
            },
            "audio": {
                "voice_type": self.voice_type,
                "encoding": "mp3",
                "speed_ratio": 1.0,
                "volume_ratio": 1.0,
                "pitch_ratio": 1.0,
            },
            "request": {
                "reqid": str(uuid.uuid4()),
                "text": text,
                "text_type": "plain",
                "operation": "query",
                "with_frontend": 1,
                "frontend_type": "unitTson"
            }
        }

        try:
            resp = requests.post(self.api_url, json.dumps(request_json), headers=self.header)
            # print(f"Response body: \n{resp.json()}")
            if "data" in resp.json():
                data = resp.json()["data"]
                with open(output_file, "wb") as file_to_save:
                    file_to_save.write(base64.b64decode(data))
                # print(f"Audio saved to {output_file}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    load_dotenv()

    appid = os.getenv("DOUBAO_APPID")
    access_token = os.getenv("DOUBAO_ACCESS_TOKEN")
    cluster = os.getenv("DOUBAO_CLUSTER")
    voice_type = os.getenv("VOICE_TYPE")

    tts = DoubaouTTS(appid, access_token, cluster, voice_type)
    tts.synthesize("Hello, this is a text to speech test.")
