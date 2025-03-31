import sys
import os
import time
import requests
import json
import threading
import queue
import numpy as np
import speech_recognition as sr
import pyttsx3
from PyQt6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                             QWidget, QLabel, QPushButton)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QPainter, QColor, QLinearGradient, QPen, QBrush, QFont

# Configuration
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2-vision:latest"
WAKE_WORD = "jarvis"

class AudioVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 150)
        self.levels = [0] * 20
        self.is_listening = False
        self.gradient_offset = 0
        
        # Set up timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)  # Update every 50ms
        
    def update_animation(self):
        self.gradient_offset = (self.gradient_offset + 1) % 100
        
        if self.is_listening:
            # Generate random levels when listening
            self.levels = [np.random.randint(10, 80) if np.random.random() > 0.2 else 
                          np.random.randint(30, 100) for _ in range(20)]
        else:
            # Gentle wave when idle
            self.levels = [20 + 10 * np.sin(time.time() * 2 + i/2) for i in range(20)]
            
        self.update()
        
    def set_listening(self, is_listening):
        self.is_listening = is_listening
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Create gradient based on state
        gradient = QLinearGradient(0, 0, self.width(), 0)
        if self.is_listening:
            gradient.setColorAt(0, QColor(41, 128, 185))  # Blue
            gradient.setColorAt(0.5, QColor(142, 68, 173))  # Purple
            gradient.setColorAt(1, QColor(41, 128, 185))  # Blue
        else:
            gradient.setColorAt(0, QColor(52, 73, 94))  # Dark Blue-Gray
            gradient.setColorAt(1, QColor(44, 62, 80))  # Darker Blue-Gray
            
        # Animate gradient
        pos = self.gradient_offset / 100
        gradient.setColorAt((pos + 0.33) % 1, QColor(52, 152, 219))  # Light Blue
        
        # Draw background
        painter.fillRect(self.rect(), QBrush(QColor(0, 10, 20)))
        
        # Draw visualizer bars
        bar_width = self.width() / len(self.levels)
        bar_margin = bar_width * 0.1
        
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(gradient))
        
        for i, level in enumerate(self.levels):
            height = max(2, int(level * self.height() / 100))
            x = i * bar_width
            y = (self.height() - height) // 2
            painter.drawRoundedRect(int(x + bar_margin), 
                                    y, 
                                    int(bar_width - 2 * bar_margin), 
                                    height, 
                                    5, 5)

class SignalCommunicator(QObject):
    status_signal = pyqtSignal(str)
    listening_signal = pyqtSignal(bool)
    thinking_signal = pyqtSignal(bool)

class AIAssistant(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_speech_engine()
        
        # Thread-safe queue for managing commands
        self.command_queue = queue.Queue()
        
        # Signal communicator for thread-safe UI updates
        self.communicator = SignalCommunicator()
        self.communicator.status_signal.connect(self.update_status)
        self.communicator.listening_signal.connect(self.set_listening)
        self.communicator.thinking_signal.connect(self.set_thinking)
        
        # Start the background thread for handling commands
        self.running = True
        self.worker_thread = threading.Thread(target=self.command_processor, daemon=True)
        self.worker_thread.start()
        
    def init_ui(self):
        self.setWindowTitle("Jarvis AI Assistant")
        self.setMinimumSize(700, 400)
        
        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Arial", 14))
        self.status_label.setStyleSheet("color: white;")
        layout.addWidget(self.status_label)
        
        # Audio visualizer
        self.visualizer = AudioVisualizer()
        layout.addWidget(self.visualizer)
        
        # Thinking indicator
        self.thinking_label = QLabel("Thinking...")
        self.thinking_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thinking_label.setFont(QFont("Arial", 12))
        self.thinking_label.setStyleSheet("color: #3498db;")
        self.thinking_label.setVisible(False)
        layout.addWidget(self.thinking_label)
        
        # Control button
        self.listen_button = QPushButton("Start Listening")
        self.listen_button.setFont(QFont("Arial", 12))
        self.listen_button.setStyleSheet("""
            QPushButton {
                background-color: #2980b9;
                color: white;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QPushButton:pressed {
                background-color: #1c6ea4;
            }
        """)
        self.listen_button.clicked.connect(self.toggle_listening)
        layout.addWidget(self.listen_button)
        
        # Set window style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a192f;
            }
            QWidget {
                background-color: #0a192f;
            }
        """)
        
        # State variables
        self.is_listening = False
        
    def init_speech_engine(self):
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        if voices:
            # Try to find a more natural-sounding voice
            for voice in voices:
                if "natural" in voice.name.lower() or "premium" in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
            # If no "natural" voice found, use the first voice
            if not self.engine.getProperty('voice'):
                self.engine.setProperty('voice', voices[0].id)
                
        # Adjust speech properties
        self.engine.setProperty('rate', 180)  # Speed
        self.engine.setProperty('volume', 0.9)  # Volume
        
        # Calibrate the recognizer for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
    
    def toggle_listening(self):
        if not self.is_listening:
            self.is_listening = True
            self.listen_button.setText("Stop Listening")
            self.command_queue.put(("start_listening", None))
        else:
            self.is_listening = False
            self.listen_button.setText("Start Listening")
            self.command_queue.put(("stop_listening", None))
    
    def command_processor(self):
        """Background thread that processes commands from the queue"""
        continuous_listening = False
        
        while self.running:
            try:
                command, data = self.command_queue.get(timeout=0.5)
                
                if command == "start_listening":
                    continuous_listening = True
                    # Start continuous listening in a separate thread
                    listen_thread = threading.Thread(
                        target=self.continuous_listen, 
                        args=(continuous_listening,)
                    )
                    listen_thread.daemon = True
                    listen_thread.start()
                    
                elif command == "stop_listening":
                    continuous_listening = False
                    
                elif command == "process_speech":
                    text = data
                    if text:
                        # Check for wake word
                        if WAKE_WORD.lower() in text.lower() or continuous_listening:
                            # Remove wake word
                            text = text.lower().replace(WAKE_WORD.lower(), "").strip()
                            if text:
                                self.communicator.thinking_signal.emit(True)
                                self.communicator.status_signal.emit(f"Processing: {text}")
                                
                                # Get response from Ollama
                                response = self.get_ollama_response(text)
                                
                                # Speak the response
                                self.communicator.thinking_signal.emit(False)
                                self.communicator.status_signal.emit(f"Responding: {response}")
                                self.speak(response)
                
                self.command_queue.task_done()
                
            except queue.Empty:
                pass  # Queue is empty, continue
                
            except Exception as e:
                print(f"Error in command processor: {e}")
    
    def continuous_listen(self, should_run):
        """Continuously listens for voice commands"""
        while should_run and self.is_listening:
            try:
                self.communicator.listening_signal.emit(True)
                self.communicator.status_signal.emit("Listening...")
                
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                
                self.communicator.status_signal.emit("Processing speech...")
                
                try:
                    text = self.recognizer.recognize_google(audio)
                    self.command_queue.put(("process_speech", text))
                except sr.UnknownValueError:
                    self.communicator.status_signal.emit("Listening...")
                except sr.RequestError as e:
                    self.communicator.status_signal.emit(f"Error: {e}")
                    
            except Exception as e:
                self.communicator.status_signal.emit(f"Error listening: {e}")
                time.sleep(1)  # Prevent tight loop in case of repeated errors
        
        self.communicator.listening_signal.emit(False)
        self.communicator.status_signal.emit("Ready")
    
    def get_ollama_response(self, prompt):
        """Get a response from the Ollama Phi model"""
        try:
            payload = {
                "model": MODEL_NAME,
                "prompt": f"You are Jarvis, a helpful AI assistant. Answer briefly and concisely. User query: {prompt}",
                "stream": False
            }
            
            response = requests.post(OLLAMA_ENDPOINT, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "I'm sorry, I couldn't process that request.")
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error communicating with the AI model: {e}"
    
    def speak(self, text):
        """Convert text to speech"""
        def _speak():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
                self.communicator.status_signal.emit("Ready")
            except Exception as e:
                self.communicator.status_signal.emit(f"Speech error: {e}")
        
        # Run in separate thread to prevent UI blocking
        speak_thread = threading.Thread(target=_speak)
        speak_thread.daemon = True
        speak_thread.start()
    
    def update_status(self, status):
        """Update the status label"""
        self.status_label.setText(status)
    
    def set_listening(self, is_listening):
        """Update listening state indicators"""
        self.visualizer.set_listening(is_listening)
    
    def set_thinking(self, is_thinking):
        """Update thinking state indicators"""
        self.thinking_label.setVisible(is_thinking)
    
    def closeEvent(self, event):
        """Clean up when window is closed"""
        self.running = False
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AIAssistant()
    window.show()
    sys.exit(app.exec())
