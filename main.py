import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import pyaudio
import wave
import numpy as np
import librosa
import os
from pathlib import Path
import subprocess
import time
import warnings
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

warnings.filterwarnings("ignore")

class SoundTriggerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sound Trigger v1.0")
        self.root.geometry("600x650")
        self.root.configure(bg='#2b2b2b')
        
        self.target_audio_path = tk.StringVar()
        self.exe_path = tk.StringVar()
        self.is_listening = False
        self.sound_model = None
        self.threshold = 2.5
        self.min_volume = tk.DoubleVar(value=0.008)
        self.last_trigger_time = 0
        self.cooldown = 1.2
        self.trigger_count = 0
        
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
            'danger': '#F44336',
            'trigger': '#FF5252'
        }
        
        self.style.configure('TFrame', background=self.colors['bg'])
        self.style.configure('TLabel', background=self.colors['bg'], foreground=self.colors['fg'])
        self.style.configure('TButton', background=self.colors['accent'], foreground=self.colors['fg'])
        self.style.configure('Header.TLabel', font=('Arial', 16, 'bold'))
        self.style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
        self.style.map('TButton', background=[('active', '#357abd')])
        
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
        
        settings_frame = ttk.LabelFrame(main_container, text="Настройки")
        settings_frame.pack(fill='x', pady=(0, 15))
        
        settings_content = ttk.Frame(settings_frame)
        settings_content.pack(padx=10, pady=10)
        
        ttk.Label(settings_content, text="Минимальная громкость:").grid(row=0, column=0, sticky='w', pady=5)
        volume_scale = ttk.Scale(settings_content, from_=0.001, to=0.05, 
                                variable=self.min_volume, orient='horizontal',
                                command=self.update_volume_label)
        volume_scale.grid(row=0, column=1, padx=(10, 0), sticky='ew')
        settings_content.columnconfigure(1, weight=1)
        
        self.volume_label = ttk.Label(settings_content, text=f"Значение: {self.min_volume.get():.3f}")
        self.volume_label.grid(row=0, column=2, padx=(10, 0))
        
        control_frame = ttk.Frame(main_container)
        control_frame.pack(fill='x', pady=20)
        
        self.status_label = ttk.Label(control_frame, text="Статус: Остановлен", 
                                     foreground=self.colors['warning'], font=('Arial', 11, 'bold'))
        self.status_label.pack(pady=(0, 10))
        
        self.listen_btn = ttk.Button(control_frame, text="Начать прослушивание", 
                                    command=self.toggle_listening, width=25)
        self.listen_btn.pack()
        
        threshold_frame = ttk.LabelFrame(main_container, text="Порог различия")
        threshold_frame.pack(fill='x', pady=(0, 15))
        
        threshold_content = ttk.Frame(threshold_frame)
        threshold_content.pack(padx=10, pady=10)
        
        self.threshold_scale = ttk.Scale(threshold_content, from_=50, to=500, 
                                        value=self.threshold, orient='horizontal',
                                        command=self.update_threshold_label)
        self.threshold_scale.pack(fill='x')
        
        self.threshold_label = ttk.Label(threshold_content, text=f"Текущее значение: {self.threshold:.2f}")
        self.threshold_label.pack(pady=(5, 0))
        
        ttk.Label(threshold_content, text="(БОЛЬШЕ значение = выше чувствительность)", 
                 foreground='#FF5252', font=('Arial', 9, 'bold')).pack()
        
        test_frame = ttk.LabelFrame(main_container, text="Тестирование")
        test_frame.pack(fill='x', pady=(0, 15))
        
        test_content = ttk.Frame(test_frame)
        test_content.pack(padx=10, pady=10)
        
        ttk.Button(test_content, text="Проверить текущий звук", 
                  command=self.test_current_sound, width=25).pack(pady=(0, 5))
        
        self.test_result = ttk.Label(test_content, text="", foreground='#aaaaaa', font=('Arial', 10))
        self.test_result.pack(pady=(5, 0))
        
        stats_frame = ttk.LabelFrame(main_container, text="Статистика")
        stats_frame.pack(fill='x', pady=(0, 15))
        
        stats_content = ttk.Frame(stats_frame)
        stats_content.pack(padx=10, pady=10)
        
        self.trigger_count_label = ttk.Label(stats_content, text="Срабатываний: 0", 
                                           foreground=self.colors['accent'])
        self.trigger_count_label.pack()
        
        info_frame = ttk.Frame(main_container)
        info_frame.pack(fill='x', pady=10)
        
        self.info_label = ttk.Label(info_frame, text="Загрузите звук-триггер и выберите приложение", 
                                   foreground='#aaaaaa', wraplength=500)
        self.info_label.pack()
        
        hint_frame = ttk.Frame(self.root)
        hint_frame.pack(fill='x', padx=30, pady=(0, 15))
        
        hint_label = ttk.Label(hint_frame, text="💡 Совет: Используйте короткий четкий звук (хлопок, щелчок). Для слабых звуков увеличьте порог до 2.5-3.5", 
                              foreground='#777777', font=('Arial', 9))
        hint_label.pack()
        
    def setup_audio(self):
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
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
    
    def update_volume_label(self, value):
        self.volume_label.config(text=f"Значение: {float(value):.3f}")
            
    def record_sound(self):
        def record():
            try:
                p = pyaudio.PyAudio()
                
                stream = p.open(format=self.FORMAT,
                              channels=self.CHANNELS,
                              rate=self.RATE,
                              input=True,
                              frames_per_buffer=self.CHUNK)
                
                self.root.after(0, lambda: messagebox.showinfo("Запись", "Запись начнется через 2 секунды.\nИздайте четкий звук (хлопок, щелчок пальцами)!"))
                time.sleep(2.5)
                
                self.root.after(0, lambda: self.status_label.config(
                    text="Запись звука...", foreground=self.colors['warning']))
                
                frames = []
                total_chunks = int(self.RATE / self.CHUNK * self.RECORD_SECONDS)
                
                for i in range(total_chunks):
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    frames.append(data)
                
                stream.stop_stream()
                stream.close()
                p.terminate()
                
                self.root.after(0, lambda: self.status_label.config(
                    text="Статус: Остановлен", foreground=self.colors['warning']))
                
                filename = filedialog.asksaveasfilename(defaultextension=".wav",
                                                   filetypes=[("WAV файлы", "*.wav")],
                                                   initialfile="trigger_sound.wav")
                if filename:
                    wf = wave.open(filename, 'wb')
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(p.get_sample_size(self.FORMAT))
                    wf.setframerate(self.RATE)
                    wf.writeframes(b''.join(frames))
                    wf.close()
                    
                    self.root.after(0, lambda: self.target_audio_path.set(filename))
                    self.root.after(0, self.train_model)
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка записи: {str(e)}"))
                self.root.after(0, lambda: self.status_label.config(
                    text="Статус: Остановлен", foreground=self.colors['warning']))
        
        self.is_listening = False
        threading.Thread(target=record, daemon=True).start()
        
    def test_current_sound(self):
        if self.sound_model is None:
            messagebox.showwarning("Внимание", "Сначала загрузите звук-триггер!")
            return
            
        def record_test():
            try:
                p = pyaudio.PyAudio()
                
                stream = p.open(format=self.FORMAT,
                              channels=self.CHANNELS,
                              rate=self.RATE,
                              input=True,
                              frames_per_buffer=self.CHUNK)
                
                self.root.after(0, lambda: messagebox.showinfo("Тест", "Издайте звук для теста (2 секунды)..."))
                time.sleep(0.5)
                
                frames = []
                for _ in range(0, int(self.RATE / self.CHUNK * 2)):
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    frames.append(data)
                
                stream.stop_stream()
                stream.close()
                p.terminate()
                
                test_audio = b''.join(frames)
                distance = self.compare_audio(test_audio, is_test=True)
                
                self.root.after(0, lambda: self.test_result.config(
                    text=f"Расстояние: {distance:.2f} (порог: {self.threshold:.2f})"))
                
                if distance < self.threshold:
                    self.root.after(0, lambda: self.test_result.config(foreground=self.colors['success']))
                    self.root.after(0, lambda: self.info_label.config(
                        text="✅ Звук распознан! Попробуйте в реальном режиме прослушивания.", 
                        foreground=self.colors['success']))
                else:
                    self.root.after(0, lambda: self.test_result.config(foreground=self.colors['warning']))
                    self.root.after(0, lambda: self.info_label.config(
                        text=f"❌ Звук не распознан. Расстояние {distance:.2f} > порога {self.threshold:.2f}. Увеличьте порог или запишите более четкий звук.", 
                        foreground=self.colors['warning']))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка теста: {str(e)}"))
        
        threading.Thread(target=record_test, daemon=True).start()
    
    def has_sufficient_volume(self, audio_data):
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        
        if len(audio_array) == 0:
            return False
            
        rms = np.sqrt(np.mean(audio_array**2))
        max_possible = 32767.0
        normalized_volume = rms / max_possible
        
        return normalized_volume > self.min_volume.get()
    
    def extract_mfcc(self, y):
        y = librosa.util.normalize(y)
        
        n_fft = 512
        hop_length = 256
        
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=self.RATE, 
            n_mfcc=13,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        return features.T
            
    def train_model(self):
        if not self.target_audio_path.get():
            return
            
        try:
            self.info_label.config(text="Обучение модели...", foreground='#aaaaaa')
            self.root.update()
            
            y, sr = librosa.load(self.target_audio_path.get(), sr=self.RATE)
            
            if len(y) < self.RATE * 0.3:
                self.info_label.config(text="Звук слишком короткий (минимум 0.3 сек)", foreground=self.colors['warning'])
                return
            
            mfcc_features = self.extract_mfcc(y)
            
            self.sound_model = {
                'mfcc': mfcc_features,
                'path': self.target_audio_path.get(),
                'length': len(y) / self.RATE
            }
            
            self.info_label.config(
                text=f"Модель обучена: {Path(self.target_audio_path.get()).name}\nДлительность: {self.sound_model['length']:.2f} сек, Фреймов: {self.sound_model['mfcc'].shape[0]}", 
                foreground=self.colors['success'])
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось обработать аудиофайл:\n{str(e)}")
            self.info_label.config(text="Ошибка при обучении модели", foreground=self.colors['danger'])
            
    def compare_audio(self, audio_data, is_test=False):
        if self.sound_model is None:
            return float('inf')
            
        try:
            if not self.has_sufficient_volume(audio_data):
                return float('inf')
            
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            if len(audio_array) < self.RATE * 0.4:
                return float('inf')
            
            current_mfcc = self.extract_mfcc(audio_array)
            ref_mfcc = self.sound_model['mfcc']
            
            if current_mfcc.shape[0] < 8 or ref_mfcc.shape[0] < 8:
                return float('inf')
            
            distance, _ = fastdtw(current_mfcc, ref_mfcc, dist=euclidean)
            normalized_distance = distance / ref_mfcc.shape[0]
            
            return normalized_distance
                
        except Exception:
            return float('inf')
            
    def listen_audio(self):
        try:
            p = pyaudio.PyAudio()
            
            stream = p.open(format=self.FORMAT,
                           channels=self.CHANNELS,
                           rate=self.RATE,
                           input=True,
                           frames_per_buffer=self.CHUNK)
            
            self.root.after(0, lambda: self.status_label.config(
                text="Статус: Прослушивание...", foreground=self.colors['success']))
            
            buffer = []
            buffer_duration = 1.2
            buffer_size = int(self.RATE / self.CHUNK * buffer_duration)
            
            while self.is_listening:
                try:
                    audio_data = stream.read(self.CHUNK, exception_on_overflow=False)
                    
                    buffer.append(audio_data)
                    if len(buffer) > buffer_size:
                        buffer.pop(0)
                    
                    if len(buffer) == buffer_size:
                        combined_audio = b''.join(buffer)
                        
                        if not self.has_sufficient_volume(combined_audio):
                            continue
                        
                        distance = self.compare_audio(combined_audio)
                        
                        current_time = time.time()
                        if distance < self.threshold and (current_time - self.last_trigger_time) > self.cooldown:
                            self.last_trigger_time = current_time
                            self.trigger_count += 1
                            self.root.after(0, lambda d=distance: self.trigger_action(d))
                            self.root.after(0, self.visual_feedback)
                            
                except Exception:
                    if self.is_listening:
                        continue
                    else:
                        break
                        
        except Exception:
            pass
        finally:
            try:
                stream.stop_stream()
                stream.close()
                p.terminate()
            except:
                pass
            
            if not self.is_listening:
                self.root.after(0, lambda: self.status_label.config(
                    text="Статус: Остановлен", foreground=self.colors['warning']))
    
    def visual_feedback(self):
        original_bg = self.root.cget('bg')
        self.root.configure(bg=self.colors['trigger'])
        self.status_label.config(text="✅ СИГНАЛ ОБНАРУЖЕН!", foreground=self.colors['success'])
        self.trigger_count_label.config(text=f"Срабатываний: {self.trigger_count}")
        
        def restore():
            self.root.configure(bg=original_bg)
            if self.is_listening:
                self.status_label.config(text="Статус: Прослушивание...", foreground=self.colors['success'])
        
        self.root.after(400, restore)
            
    def trigger_action(self, distance):
        exe_path = self.exe_path.get().strip()
        if not exe_path:
            self.info_label.config(text="Ошибка: не указан путь к приложению", foreground=self.colors['danger'])
            return
            
        try:
            if not os.path.exists(exe_path):
                self.info_label.config(text=f"Ошибка: файл не найден:\n{exe_path}", foreground=self.colors['danger'])
                return
                
            self.info_label.config(text=f"Запуск приложения... (расстояние: {distance:.2f})", foreground=self.colors['success'])
            
            try:
                subprocess.Popen([exe_path], shell=True)
            except Exception as e1:
                try:
                    subprocess.Popen(exe_path, shell=True)
                except Exception as e2:
                    raise Exception(f"{str(e1)} / {str(e2)}")
            
            self.root.after(2500, lambda: self.info_label.config(
                text=f"✅ Приложение запущено успешно!", 
                foreground=self.colors['success']))
                
        except Exception as e:
            error_msg = f"Ошибка запуска:\n{str(e)[:80]}"
            self.info_label.config(text=error_msg, foreground=self.colors['danger'])
            self.root.after(3000, lambda: self.info_label.config(
                text="Попробуйте выбрать другой EXE файл", foreground='#aaaaaa'))
                
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
            self.listen_btn.config(text="⏹ Остановить прослушивание")
            self.listen_thread = threading.Thread(target=self.listen_audio, daemon=True)
            self.listen_thread.start()
        else:
            self.listen_btn.config(text="▶ Начать прослушивание")
            self.status_label.config(text="Статус: Остановлен", foreground=self.colors['warning'])

def main():
    try:
        import pyaudio
        import librosa
        from fastdtw import fastdtw
    except ImportError as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Ошибка", 
            "Не установлены необходимые библиотеки:\n\n"
            "Установите командой:\n"
            "pip install pyaudio librosa numpy fastdtw scipy\n\n"
            f"Ошибка: {str(e)}")
        return
        
    root = tk.Tk()
    app = SoundTriggerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()