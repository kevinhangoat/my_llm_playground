import pyaudio
import webrtcvad
import collections
import speech_recognition as sr
import numpy as np
import whisper
import wave
import threading
import os

class LiveSpeechToText:
    def __init__(self, vad_aggressiveness=3):
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.tmp_result = None
        
        self.delete_audio_file()
        self.model = sr.Recognizer()
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=16000,
                                      input=True,
                                      frames_per_buffer=1024)
        self.stream.start_stream()

    def read_audio(self):
        return self.stream.read(320)

    def recognize_speech(self):
        self.delete_audio_file()
        while not self.stop_event.is_set():
            audio_data = b''.join([self.read_audio() for _ in range(int(5 * 16000 / 320))])
            self._save_audio(audio_data)
            with sr.AudioFile("output.wav") as source:
                audio = self.model.record(source)
            try:
                text = self.model.recognize_google(audio, language="en-US")
                # print(text)
            except sr.UnknownValueError:
                try:
                    text = self.model.recognize_google(audio, language="zh-CN")
                    # print(text)
                except sr.UnknownValueError:
                    # print("Google Speech Recognition could not understand audio")
                    continue
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))
                    continue
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
                continue
        self.tmp_result = text

    def start_recognition(self):
        self.stop_event = threading.Event()
        listener_thread = threading.Thread(target=self.recognize_speech)
        listener_thread.start()

        input("Press Enter to stop...\n")
        self.stop_event.set()
        listener_thread.join()
        return self.tmp_result
    
    def delete_audio_file(self):
        filename = "output.wav"
        if os.path.exists(filename):
            os.remove(filename)
        else:
            print("The file does not exist")
            
    def _save_audio(self, audio_data):
        filename = "output.wav"
        try:
            with wave.open(filename, 'rb') as wf:
                params = wf.getparams()
                existing_data = wf.readframes(wf.getnframes())
        except FileNotFoundError:
            params = (1, 2, 16000, 0, 'NONE', 'not compressed')
            existing_data = b''

        with wave.open(filename, 'wb') as wf:
            wf.setparams(params)
            wf.writeframes(existing_data)
            wf.writeframes(audio_data)

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
            return result["text"]

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

if __name__ == "__main__":
    live_stt = LiveSpeechToText()
    try:
        live_stt.start_recognition()
        text = live_stt.recognize_speech()
    except KeyboardInterrupt:
        pass
    finally:
        live_stt.close()