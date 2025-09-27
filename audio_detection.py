import pyaudio
import numpy as np
import threading
import time
from collections import deque
import speech_recognition as sr
from scipy.fft import fft
import warnings
warnings.filterwarnings("ignore")

class AudioDetector:
    def __init__(self, 
                 chunk_size=1024,
                 sample_rate=44100,
                 threshold_db=15,
                 speech_threshold=0.1,
                 suspicious_words=None):
        
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.threshold_db = threshold_db
        self.speech_threshold = speech_threshold
        
        # Suspicious keywords that might indicate cheating
        if suspicious_words is None:
            self.suspicious_words = [
                'answer', 'help', 'question', 'tell me', 'what is',
                'google', 'search', 'phone', 'call', 'message',
                'copy', 'paste', 'share', 'send', 'receive'
            ]
        else:
            self.suspicious_words = suspicious_words
            
        # Audio monitoring variables
        self.is_monitoring = False
        self.audio_thread = None
        self.audio_buffer = deque(maxlen=100)
        
        # Detection results
        self.current_db_level = 0
        self.speech_detected = False
        self.suspicious_speech = False
        self.last_suspicious_text = ""
        
        # Initialize PyAudio
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = None
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Calibrate for ambient noise
            print("Calibrating microphone for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Microphone calibrated.")
            
        except Exception as e:
            print(f"Error initializing audio: {e}")
            self.audio = None
    
    def calculate_db_level(self, audio_data):
        """Calculate decibel level from audio data"""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate RMS (Root Mean Square)
            rms = np.sqrt(np.mean(audio_array**2))
            
            # Convert to decibels (with protection against log(0))
            if rms > 0:
                db_level = 20 * np.log10(rms / 32767.0)  # Normalize to max int16 value
                return max(0, db_level + 96)  # Adjust to positive scale
            else:
                return 0
        except:
            return 0
    
    def detect_speech_energy(self, audio_data):
        """Detect if audio contains speech-like energy patterns"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            # Apply FFT to get frequency components
            fft_data = fft(audio_array)
            freqs = np.fft.fftfreq(len(fft_data), 1/self.sample_rate)
            
            # Focus on speech frequency range (300-3400 Hz)
            speech_range = (freqs >= 300) & (freqs <= 3400)
            speech_energy = np.mean(np.abs(fft_data[speech_range]))
            
            # Normalize and check if above threshold
            total_energy = np.mean(np.abs(fft_data))
            
            if total_energy > 0:
                speech_ratio = speech_energy / total_energy
                return speech_ratio > self.speech_threshold
            
            return False
        except:
            return False
    
    def recognize_speech_from_buffer(self):
        """Convert recent audio buffer to text using speech recognition"""
        try:
            if len(self.audio_buffer) < 10:  # Need enough audio data
                return None
            
            # Combine recent audio chunks
            combined_audio = b''.join(list(self.audio_buffer)[-20:])  # Last 20 chunks
            
            # Convert to audio format that speech_recognition can use
            audio_array = np.frombuffer(combined_audio, dtype=np.int16)
            
            # Create AudioData object
            audio_data = sr.AudioData(combined_audio, self.sample_rate, 2)
            
            # Recognize speech
            text = self.recognizer.recognize_google(audio_data, language='en-US')
            return text.lower()
            
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None
        except Exception as e:
            return None
    
    def check_suspicious_words(self, text):
        """Check if detected speech contains suspicious words"""
        if not text:
            return False, ""
        
        detected_words = []
        for word in self.suspicious_words:
            if word in text:
                detected_words.append(word)
        
        if detected_words:
            return True, f"Detected: {', '.join(detected_words)}"
        
        return False, ""
    
    def audio_monitoring_thread(self):
        """Main audio monitoring thread"""
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            speech_detection_counter = 0
            last_speech_recognition = time.time()
            
            while self.is_monitoring:
                try:
                    # Read audio data
                    audio_data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    self.audio_buffer.append(audio_data)
                    
                    # Calculate decibel level
                    self.current_db_level = self.calculate_db_level(audio_data)
                    
                    # Detect speech-like patterns
                    has_speech_energy = self.detect_speech_energy(audio_data)
                    
                    if has_speech_energy and self.current_db_level > self.threshold_db:
                        speech_detection_counter += 1
                        self.speech_detected = True
                        
                        # Try speech recognition every 2 seconds when speech is detected
                        current_time = time.time()
                        if current_time - last_speech_recognition > 2.0:
                            recognized_text = self.recognize_speech_from_buffer()
                            if recognized_text:
                                is_suspicious, suspicious_details = self.check_suspicious_words(recognized_text)
                                self.suspicious_speech = is_suspicious
                                if is_suspicious:
                                    self.last_suspicious_text = suspicious_details
                                    print(f"Suspicious speech detected: {recognized_text}")
                            
                            last_speech_recognition = current_time
                    else:
                        speech_detection_counter = max(0, speech_detection_counter - 1)
                        if speech_detection_counter == 0:
                            self.speech_detected = False
                    
                    time.sleep(0.01)  # Small delay to prevent excessive CPU usage
                    
                except Exception as e:
                    print(f"Error in audio monitoring: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error starting audio stream: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
    
    def start_monitoring(self):
        """Start audio monitoring in a separate thread"""
        if self.audio is None:
            print("Audio system not available")
            return False
        
        if self.is_monitoring:
            return True
        
        self.is_monitoring = True
        self.audio_thread = threading.Thread(target=self.audio_monitoring_thread)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        return True
    
    def stop_monitoring(self):
        """Stop audio monitoring"""
        self.is_monitoring = False
        if self.audio_thread:
            self.audio_thread.join(timeout=1)
    
    def get_audio_status(self):
        """Get current audio detection status"""
        return {
            'db_level': round(self.current_db_level, 1),
            'speech_detected': self.speech_detected,
            'suspicious_speech': self.suspicious_speech,
            'suspicious_text': self.last_suspicious_text
        }
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_monitoring()
        if self.audio:
            self.audio.terminate()

# Usage example and testing functions
if __name__ == "__main__":
    # Test the audio detector
    detector = AudioDetector(threshold_db=20, speech_threshold=0.1)
    
    # Test basic audio
    print("Testing if we can hear audio...")
    for i in range(5):
        status = detector.get_audio_status()
        print(f"Sound level: {status['db_level']}")
        time.sleep(1)
    
    if detector.start_monitoring():
        print("Audio monitoring started. Speak to test detection...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                status = detector.get_audio_status()
                print(f"\rDB Level: {status['db_level']}, "
                      f"Speech: {status['speech_detected']}, "
                      f"Suspicious: {status['suspicious_speech']} "
                      f"{status['suspicious_text']}", end="")
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nStopping audio monitoring...")
            detector.cleanup()
    else:
        print("Failed to start audio monitoring")
