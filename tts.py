import pyttsx3

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

if __name__ == "__main__":
    tts = TextToSpeech()
    tts.speak("Hello, this is a text to speech test.")