import pyaudio
import webrtcvad
import collections
import speech_recognition as sr
import numpy as np
import whisper

class LiveSpeechToText:
    def __init__(self, vad_aggressiveness=3):
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.recognizer = sr.Recognizer()
        self.audio = pyaudio.PyAudio()
        # self.stream = self.audio.open(format=pyaudio.paInt16,
        #                               channels=1,
        #                               rate=16000,
        #                               input=True,
        #                               frames_per_buffer=320)
        # self.stream.start_stream()

    def read_audio(self):
        return self.stream.read(320)

    def vad_collector(self, padding_ms=300, ratio=0.75):
        num_padding_frames = int(padding_ms / 30)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        voiced_frames = []
        while True:
            frame = self.read_audio()
            is_speech = self.vad.is_speech(frame, 16000)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    voiced_frames.extend([f for f, s in ring_buffer])
                    ring_buffer.clear()
            else:
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield b''.join(voiced_frames)
                    ring_buffer.clear()
                    voiced_frames = []

    def recognize_speech(self):
        for audio_data in self.vad_collector():
            audio = sr.AudioData(audio_data, 16000, 2)
            try:
                text = self.recognizer.recognize_google(audio, language="en-US")
                return text
            except sr.UnknownValueError:
                try:
                    text = self.recognizer.recognize_google(audio, language="zh-CN")
                    return text
                except sr.UnknownValueError:
                    # print("Google Speech Recognition could not understand audio")
                    continue
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))
                    continue
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
                continue

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

class WhisperLiveSpeechToText:
    def __init__(self, vad_aggressiveness=3):
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.model = whisper.load_model("base")
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=16000,
                                      input=True,
                                      frames_per_buffer=320)
        self.stream.start_stream()

    def read_audio(self):
        return self.stream.read(320)

    def vad_collector(self, padding_ms=900, ratio=0.75):
        num_padding_frames = int(padding_ms / 30)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        voiced_frames = []
        while True:
            frame = self.read_audio()
            is_speech = self.vad.is_speech(frame, 16000)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    voiced_frames.extend([f for f, s in ring_buffer])
                    ring_buffer.clear()
            else:
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield b''.join(voiced_frames)
                    ring_buffer.clear()
                    voiced_frames = []

    def recognize_speech(self):
        for audio_data in self.vad_collector():
            audio_array = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0
            result = self.model.transcribe(audio_array)
            print("You said: " + result["text"])

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

if __name__ == "__main__":
    live_stt = WhisperLiveSpeechToText()
    try:
        text = live_stt.recognize_speech()
    except KeyboardInterrupt:
        pass
    finally:
        live_stt.close()