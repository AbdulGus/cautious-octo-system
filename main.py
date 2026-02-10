import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import pyaudio
import wave
import numpy as np
import librosa
import os
from pathlib import Path
import subprocess
import time
import warnings

warnings.filterwarnings("ignore")

class SoundTriggerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sound Trigger v1.0")
        self.root.geometry("600x500")
        self.root.configure(bg='#2b2b2b')
        
        self.target_audio_path = tk.StringVar()
        self.exe_path = tk.StringVar()
        self.is_listening = False
        self.sound_model = None
        self.threshold = 0.7
        self.audio_queue = queue.Queue()
        self.last_trigger_time = 0
        self.cooldown = 2.0
        
        self.setup_styles()
        self.setup_ui()
        self.setup_audio()
        
    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        self.colors = {
            'bg': '#2b2b2b',
            'fg': '#ffffff',
            'accent': '#4a90e2',
            'secondary': '#3c3c3c',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'danger': '#F44336'
        }
        
        self.style.configure('TFrame', background=self.colors['bg'])
        self.style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['fg'])
        self.style.configure('TButton', background=self.colors['accent'], foreground=self.colors['fg'])
        self.style.configure('Header.TLabel', font=('Arial', 16, 'bold'))
        self.style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
        
    def setup_ui(self):
        header_frame = ttk.Frame(self.root)
        header_frame.pack(pady=20)
        
        title_label = ttk.Label(header_frame, text="🎵 Sound Trigger", style='Header.TLabel')
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame, text="Запускайте приложения звуковыми командами", 
                                 foreground='#aaaaaa')
        subtitle_label.pack()
        
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=30, pady=10)
        
        sound_frame = ttk.LabelFrame(main_container, text="Выбор звука-триггера")
        sound_frame.pack(fill='x', pady=(0, 15))
        
        sound_content = ttk.Frame(sound_frame)
        sound_content.pack(padx=10, pady=10)
        
        ttk.Label(sound_content, text="Выберите аудиофайл:").grid(row=0, column=0, sticky='w', pady=5)
        
        sound_entry = ttk.Entry(sound_content, textvariable=self.target_audio_path, width=40)
        sound_entry.grid(row=1, column=0, padx=(0, 10))
        
        ttk.Button(sound_content, text="Обзор...", 
                  command=self.browse_audio).grid(row=1, column=1)
        
        ttk.Button(sound_content, text="Записать звук", 
                  command=self.record_sound, width=15).grid(row=2, column=0, columnspan=2, pady=10)
        
        app_frame = ttk.LabelFrame(main_container, text="Выбор приложения для запуска")
        app_frame.pack(fill='x', pady=(0, 15))
        
        app_content = ttk.Frame(app_frame)
        app_content.pack(padx=10, pady=10)
        
        ttk.Label(app_content, text="Выберите EXE файл:").grid(row=0, column=0, sticky='w', pady=5)
        
        app_entry = ttk.Entry(app_content, textvariable=self.exe_path, width=40)
        app_entry.grid(row=1, column=0, padx=(0, 10))
        
        ttk.Button(app_content, text="Обзор...", 
                  command=self.browse_exe).grid(row=1, column=1)
        
        control_frame = ttk.Frame(main_container)
        control_frame.pack(fill='x', pady=20)
        
        self.status_label = ttk.Label(control_frame, text="Статус: Остановлен", 
                                     foreground=self.colors['warning'])
        self.status_label.pack(pady=(0, 10))
        
        self.listen_btn = ttk.Button(control_frame, text="Начать прослушивание", 
                                    command=self.toggle_listening, width=20)
        self.listen_btn.pack()
        
        threshold_frame = ttk.LabelFrame(main_container, text="Чувствительность распознавания")
        threshold_frame.pack(fill='x', pady=(0, 15))
        
        threshold_content = ttk.Frame(threshold_frame)
        threshold_content.pack(padx=10, pady=10)
        
        self.threshold_scale = ttk.Scale(threshold_content, from_=0.1, to=1.0, 
                                        value=self.threshold, orient='horizontal',
                                        command=self.update_threshold_label)
        self.threshold_scale.pack(fill='x')
        
        self.threshold_label = ttk.Label(threshold_content, text=f"Текущее значение: {self.threshold:.2f}")
        self.threshold_label.pack()
        
        info_frame = ttk.Frame(main_container)
        info_frame.pack(fill='x', pady=10)
        
        self.info_label = ttk.Label(info_frame, text="Загрузите звук-триггер и выберите приложение", 
                                   foreground='#aaaaaa', wraplength=500)
        self.info_label.pack()
        
    def setup_audio(self):
        self.CHUNK = 2048
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.RECORD_SECONDS = 2
        
    def browse_audio(self):
        filetypes = [("Аудио файлы", "*.wav *.mp3 *.ogg *.flac"), ("Все файлы", "*.*")]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.target_audio_path.set(filename)
            self.train_model()
            
    def browse_exe(self):
        filetypes = [("Приложения", "*.exe"), ("Все файлы", "*.*")]
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            self.exe_path.set(filename)
            
    def update_threshold_label(self, value):
        self.threshold = float(value)
        self.threshold_label.config(text=f"Текущее значение: {self.threshold:.2f}")
            
    def record_sound(self):
        def record():
            p = pyaudio.PyAudio()
            
            stream = p.open(format=self.FORMAT,
                          channels=self.CHANNELS,
                          rate=self.RATE,
                          input=True,
                          frames_per_buffer=self.CHUNK)
            
            messagebox.showinfo("Запись", "Говорите сейчас! Запись начнется через 3 секунды...")
            self.root.after(3000, lambda: self.start_recording(p, stream))
            
        threading.Thread(target=record, daemon=True).start()
        
    def start_recording(self, p, stream):
        frames = []
        
        for _ in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        filename = filedialog.asksaveasfilename(defaultextension=".wav",
                                               filetypes=[("WAV файлы", "*.wav")])
        if filename:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            
            self.target_audio_path.set(filename)
            self.train_model()
            
    def extract_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.RATE)
            
            if len(y) < self.CHUNK:
                y = np.pad(y, (0, self.CHUNK - len(y)), mode='constant')
            
            n_fft = min(2048, len(y))
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=n_fft//4)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            
            features = mfcc_mean
            
            return features, sr
        except Exception as e:
            print(f"Ошибка при извлечении признаков: {e}")
            return None, None
            
    def train_model(self):
        if not self.target_audio_path.get():
            return
            
        try:
            self.info_label.config(text="Обучение модели...")
            self.root.update()
            
            features, sr = self.extract_features(self.target_audio_path.get())
            if features is not None:
                self.sound_model = {
                    'features': features,
                    'sr': sr,
                    'path': self.target_audio_path.get()
                }
                self.info_label.config(text=f"Модель обучена: {Path(self.target_audio_path.get()).name}")
            else:
                self.info_label.config(text="Ошибка при обучении модели")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обработать аудиофайл: {e}")
            
    def compare_audio(self, audio_data):
        if self.sound_model is None:
            return 0
            
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            if len(audio_array) < self.CHUNK:
                audio_array = np.pad(audio_array, (0, self.CHUNK - len(audio_array)), mode='constant')
            
            n_fft = min(2048, len(audio_array))
            
            mfcc_current = librosa.feature.mfcc(y=audio_array, sr=self.RATE, n_mfcc=13, 
                                               n_fft=n_fft, hop_length=n_fft//4)
            mfcc_mean_current = np.mean(mfcc_current.T, axis=0)
            
            if len(mfcc_mean_current) != len(self.sound_model['features']):
                min_len = min(len(mfcc_mean_current), len(self.sound_model['features']))
                similarity = np.dot(mfcc_mean_current[:min_len], self.sound_model['features'][:min_len])
                similarity /= (np.linalg.norm(mfcc_mean_current[:min_len]) * 
                              np.linalg.norm(self.sound_model['features'][:min_len]) + 1e-10)
            else:
                similarity = np.dot(mfcc_mean_current, self.sound_model['features'])
                similarity /= (np.linalg.norm(mfcc_mean_current) * 
                              np.linalg.norm(self.sound_model['features']) + 1e-10)
            
            return abs(similarity)
        except Exception as e:
            print(f"Ошибка сравнения: {e}")
            return 0
            
    def listen_audio(self):
        p = pyaudio.PyAudio()
        
        stream = p.open(format=self.FORMAT,
                       channels=self.CHANNELS,
                       rate=self.RATE,
                       input=True,
                       frames_per_buffer=self.CHUNK)
        
        self.status_label.config(text="Статус: Прослушивание...", foreground=self.colors['success'])
        
        buffer = []
        buffer_size = int(self.RATE / self.CHUNK * 0.5)
        
        while self.is_listening:
            try:
                audio_data = stream.read(self.CHUNK, exception_on_overflow=False)
                
                buffer.append(audio_data)
                if len(buffer) > buffer_size:
                    buffer.pop(0)
                
                if len(buffer) == buffer_size:
                    combined_audio = b''.join(buffer)
                    similarity = self.compare_audio(combined_audio)
                    
                    current_time = time.time()
                    if similarity > self.threshold and (current_time - self.last_trigger_time) > self.cooldown:
                        print(f"Обнаружен звук! Сходство: {similarity:.2f}")
                        self.last_trigger_time = current_time
                        self.root.after(0, self.trigger_action)
                        
            except Exception as e:
                print(f"Ошибка при прослушивании: {e}")
                break
                
        stream.stop_stream()
        stream.close()
        p.terminate()
        
    def trigger_action(self):
        if self.exe_path.get():
            try:
                self.info_label.config(text="Запуск приложения...", foreground=self.colors['success'])
                subprocess.Popen(self.exe_path.get())
                
                self.root.after(2000, lambda: self.info_label.config(
                    text=f"Приложение запущено!",
                    foreground='#aaaaaa'
                ))
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось запустить приложение: {e}")
                self.info_label.config(text="Ошибка запуска приложения", foreground=self.colors['danger'])
                
    def toggle_listening(self):
        if not self.target_audio_path.get():
            messagebox.showwarning("Внимание", "Сначала выберите звук-триггер!")
            return
            
        if not self.exe_path.get():
            messagebox.showwarning("Внимание", "Сначала выберите приложение для запуска!")
            return
            
        if not self.sound_model:
            messagebox.showwarning("Внимание", "Сначала обучите модель звука!")
            return
            
        self.is_listening = not self.is_listening
        
        if self.is_listening:
            self.listen_btn.config(text="Остановить прослушивание")
            self.listen_thread = threading.Thread(target=self.listen_audio, daemon=True)
            self.listen_thread.start()
        else:
            self.listen_btn.config(text="Начать прослушивание")
            self.status_label.config(text="Статус: Остановлен", foreground=self.colors['warning'])

def main():
    try:
        import pyaudio
        import librosa
    except ImportError as e:
        print("Установите необходимые библиотеки:")
        print("pip install pyaudio librosa numpy")
        return
        
    root = tk.Tk()
    app = SoundTriggerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()