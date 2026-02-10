import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import pyaudio
import wave
import numpy as np
import librosa
import os
import json
import time
import uuid
from pathlib import Path
import subprocess
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

class ConfigPanel(ttk.Frame):
    def __init__(self, parent, app, config_id=None):
        super().__init__(parent, style="Config.TFrame")
        self.app = app
        self.config_id = config_id or str(uuid.uuid4())
        self.sound_model = None
        self.last_trigger = 0
        self.is_trained = False
        self.cooldown = 1.2
        self.data = {
            'name': f"Триггер {len(app.configs) + 1}",
            'sound_path': '',
            'exe_path': '',
            'threshold': 2.5,
            'min_volume': 0.008,
            'enabled': True
        }
        self.create_widgets()
        self.update_appearance()
    
    def create_widgets(self):
        header = ttk.Frame(self, style="ConfigHeader.TFrame")
        header.pack(fill='x', padx=8, pady=(8, 4))
        
        self.name_var = tk.StringVar(value=self.data['name'])
        name_entry = ttk.Entry(header, textvariable=self.name_var, width=20,
                             style="ConfigName.TEntry", font=('Segoe UI', 10, 'bold'))
        name_entry.pack(side='left', padx=(4, 8))
        name_entry.bind('<FocusOut>', self.on_name_change)
        
        self.enabled_var = tk.BooleanVar(value=self.data['enabled'])
        enable_btn = ttk.Checkbutton(header, variable=self.enabled_var,
                                   text="Активен", style="ConfigToggle.TCheckbutton",
                                   command=self.on_toggle)
        enable_btn.pack(side='left', padx=(0, 10))
        
        btn_frame = ttk.Frame(header, style="ConfigHeader.TFrame")
        btn_frame.pack(side='right')
        ttk.Button(btn_frame, text="Тест", style="ConfigAction.TButton",
                 command=self.test_trigger, width=6).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="✎", style="ConfigEdit.TButton",
                 command=self.edit_paths, width=3).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="🗑", style="ConfigDelete.TButton",
                 command=self.delete_self, width=3).pack(side='left', padx=(2, 0))
        
        content = ttk.Frame(self, style="ConfigContent.TFrame")
        content.pack(fill='both', expand=True, padx=12, pady=(0, 12))
        
        sound_frame = ttk.Frame(content, style="ConfigSection.TFrame")
        sound_frame.pack(fill='x', pady=4)
        ttk.Label(sound_frame, text="🔊 Звук-триггер", style="ConfigSectionLabel.TLabel").pack(anchor='w')
        path_frame = ttk.Frame(sound_frame, style="ConfigSection.TFrame")
        path_frame.pack(fill='x', pady=(4, 0))
        self.sound_path_var = tk.StringVar(value=self.data['sound_path'])
        path_entry = ttk.Entry(path_frame, textvariable=self.sound_path_var,
                             state='readonly', style="ConfigPath.TEntry")
        path_entry.pack(side='left', fill='x', expand=True, padx=(0, 6))
        ttk.Button(path_frame, text="Обзор", style="ConfigButton.TButton",
                 command=self.browse_sound, width=8).pack(side='left')
        ttk.Button(path_frame, text="Запись", style="ConfigRecord.TButton",
                 command=self.record_sound, width=8).pack(side='left', padx=(4, 0))
        
        exe_frame = ttk.Frame(content, style="ConfigSection.TFrame")
        exe_frame.pack(fill='x', pady=4)
        ttk.Label(exe_frame, text="🚀 Приложение", style="ConfigSectionLabel.TLabel").pack(anchor='w')
        exe_path_frame = ttk.Frame(exe_frame, style="ConfigSection.TFrame")
        exe_path_frame.pack(fill='x', pady=(4, 0))
        self.exe_path_var = tk.StringVar(value=self.data['exe_path'])
        exe_entry = ttk.Entry(exe_path_frame, textvariable=self.exe_path_var,
                            state='readonly', style="ConfigPath.TEntry")
        exe_entry.pack(side='left', fill='x', expand=True, padx=(0, 6))
        ttk.Button(exe_path_frame, text="Обзор", style="ConfigButton.TButton",
                 command=self.browse_exe, width=8).pack(side='left')
        
        sens_frame = ttk.Frame(content, style="ConfigSection.TFrame")
        sens_frame.pack(fill='x', pady=6)
        
        thresh_frame = ttk.Frame(sens_frame, style="ConfigSection.TFrame")
        thresh_frame.pack(fill='x', pady=2)
        ttk.Label(thresh_frame, text="Чувствительность", style="ConfigSliderLabel.TLabel").pack(side='left')
        self.thresh_var = tk.DoubleVar(value=self.data['threshold'])
        self.thresh_label = ttk.Label(thresh_frame, text=f"{self.thresh_var.get():.1f}",
                                    style="ConfigSliderValue.TLabel")
        self.thresh_label.pack(side='right')
        thresh_slider = ttk.Scale(thresh_frame, from_=50, to=500,
                                variable=self.thresh_var, orient='horizontal',
                                command=self.on_thresh_change, style="ConfigSlider.Horizontal.TScale")
        thresh_slider.pack(fill='x', pady=(2, 0), padx=(5, 0))
        ttk.Label(sens_frame, text="← Меньше    Больше →",
                style="ConfigHint.TLabel").pack(anchor='w', padx=5)
        
        vol_frame = ttk.Frame(sens_frame, style="ConfigSection.TFrame")
        vol_frame.pack(fill='x', pady=4)
        ttk.Label(vol_frame, text="Мин. громкость", style="ConfigSliderLabel.TLabel").pack(side='left')
        self.vol_var = tk.DoubleVar(value=self.data['min_volume'])
        self.vol_label = ttk.Label(vol_frame, text=f"{self.vol_var.get():.3f}",
                                 style="ConfigSliderValue.TLabel")
        self.vol_label.pack(side='right')
        vol_slider = ttk.Scale(vol_frame, from_=0.001, to=0.1,
                             variable=self.vol_var, orient='horizontal',
                             command=self.on_vol_change, style="ConfigSlider.Horizontal.TScale")
        vol_slider.pack(fill='x', pady=(2, 0), padx=(5, 0))
        
        self.status_label = ttk.Label(content, text="⬤ Модель не обучена",
                                    style="ConfigStatusInactive.TLabel")
        self.status_label.pack(anchor='w', pady=(6, 0))
    
    def update_appearance(self):
        if self.is_trained and self.enabled_var.get():
            self.status_label.configure(text="⬤ Готов", style="ConfigStatusActive.TLabel")
            self.configure(style="ConfigActive.TFrame")
        elif not self.enabled_var.get():
            self.status_label.configure(text="⬤ Отключён", style="ConfigStatusDisabled.TLabel")
            self.configure(style="ConfigDisabled.TFrame")
        else:
            self.status_label.configure(text="⬤ Модель не обучена", style="ConfigStatusInactive.TLabel")
            self.configure(style="Config.TFrame")
    
    def on_name_change(self, event=None):
        self.data['name'] = self.name_var.get()
    
    def on_toggle(self):
        self.data['enabled'] = self.enabled_var.get()
        self.update_appearance()
    
    def on_thresh_change(self, value):
        val = float(value)
        self.thresh_var.set(round(val, 1))
        self.thresh_label.config(text=f"{val:.1f}")
        self.data['threshold'] = val
    
    def on_vol_change(self, value):
        val = float(value)
        self.vol_var.set(round(val, 3))
        self.vol_label.config(text=f"{val:.3f}")
        self.data['min_volume'] = val
    
    def browse_sound(self):
        filetypes = [("Аудиофайлы", "*.wav *.mp3 *.ogg *.flac"), ("Все файлы", "*.*")]
        path = filedialog.askopenfilename(filetypes=filetypes, title="Выберите звук-триггер")
        if path:
            self.sound_path_var.set(path)
            self.data['sound_path'] = path
            self.train_model()
    
    def browse_exe(self):
        filetypes = [("Приложения", "*.exe"), ("Все файлы", "*.*")]
        path = filedialog.askopenfilename(filetypes=filetypes, title="Выберите приложение")
        if path:
            self.exe_path_var.set(path)
            self.data['exe_path'] = path
    
    def edit_paths(self):
        dialog = tk.Toplevel(self.app.root)
        dialog.title(f"Редактировать пути - {self.data['name']}")
        dialog.geometry("500x200")
        dialog.transient(self.app.root)
        dialog.grab_set()
        dialog.configure(bg='#1e1e1e')
        
        ttk.Label(dialog, text="Путь к звуку-триггеру:").pack(anchor='w', padx=15, pady=(15, 0))
        sound_entry = ttk.Entry(dialog, width=60)
        sound_entry.insert(0, self.data['sound_path'])
        sound_entry.pack(padx=15, pady=5, fill='x')
        
        ttk.Label(dialog, text="Путь к приложению:").pack(anchor='w', padx=15, pady=(10, 0))
        exe_entry = ttk.Entry(dialog, width=60)
        exe_entry.insert(0, self.data['exe_path'])
        exe_entry.pack(padx=15, pady=5, fill='x')
        
        def save():
            self.data['sound_path'] = sound_entry.get()
            self.data['exe_path'] = exe_entry.get()
            self.sound_path_var.set(self.data['sound_path'])
            self.exe_path_var.set(self.data['exe_path'])
            if self.data['sound_path']:
                self.train_model()
            dialog.destroy()
        
        btn_frame = ttk.Frame(dialog, style="TFrame")
        btn_frame.pack(fill='x', padx=15, pady=15)
        ttk.Button(btn_frame, text="Сохранить", command=save, style="Accent.TButton").pack(side='right', padx=5)
        ttk.Button(btn_frame, text="Отмена", command=dialog.destroy).pack(side='right', padx=5)
    
    def record_sound(self):
        self.app.record_sound_for_config(self)
    
    def train_model(self):
        path = self.data['sound_path']
        if not path or not os.path.exists(path):
            self.status_label.configure(text="⬤ Файл не выбран", style="ConfigStatusError.TLabel")
            self.is_trained = False
            self.update_appearance()
            return
        
        try:
            self.status_label.configure(text="⬤ Обучение...", style="ConfigStatusBusy.TLabel")
            self.app.root.update()
            y, sr = librosa.load(path, sr=self.app.RATE)
            if len(y) < self.app.RATE * 0.3:
                raise ValueError("Звук слишком короткий (мин. 0.3с)")
            
            y = librosa.util.normalize(y)
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256
            )
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            features = np.vstack([mfcc, mfcc_delta, mfcc_delta2]).T
            self.sound_model = {
                'mfcc': features,
                'frames': features.shape[0],
                'path': path,
                'duration': len(y) / sr
            }
            self.is_trained = True
            self.status_label.configure(text=f"⬤ Готов ({self.sound_model['duration']:.1f}с)",
                                      style="ConfigStatusActive.TLabel")
        except Exception as e:
            self.is_trained = False
            self.status_label.configure(text=f"⬤ Ошибка: {str(e)[:30]}", style="ConfigStatusError.TLabel")
        finally:
            self.update_appearance()
    
    def test_trigger(self):
        if not self.is_trained:
            messagebox.showwarning("Не готово", "Сначала обучите модель звука!")
            return
        self.app.test_single_config(self)
    
    def delete_self(self):
        if messagebox.askyesno("Подтвердите удаление", f"Удалить профиль '{self.data['name']}'?"):
            self.app.remove_config(self)
    
    def to_dict(self):
        return {
            'id': self.config_id,
            'name': self.data['name'],
            'sound_path': self.data['sound_path'],
            'exe_path': self.data['exe_path'],
            'threshold': self.data['threshold'],
            'min_volume': self.data['min_volume'],
            'enabled': self.data['enabled']
        }
    
    def from_dict(self, data):
        self.config_id = data.get('id', str(uuid.uuid4()))
        self.data.update(data)
        self.name_var.set(self.data['name'])
        self.sound_path_var.set(self.data['sound_path'])
        self.exe_path_var.set(self.data['exe_path'])
        self.thresh_var.set(self.data['threshold'])
        self.vol_var.set(self.data['min_volume'])
        self.enabled_var.set(self.data['enabled'])
        self.on_thresh_change(self.data['threshold'])
        self.on_vol_change(self.data['min_volume'])
        self.on_toggle()
        if self.data['sound_path']:
            self.train_model()

class SoundTriggerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SonicTrigger • Активатор приложений по звуку")
        self.root.geometry("800x750")
        self.root.minsize(750, 600)
        self.root.configure(bg='#1a1a1a')
        
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.BUFFER_DURATION = 1.0
        self.BUFFER_SIZE = int(self.RATE / self.CHUNK * self.BUFFER_DURATION)
        
        self.is_listening = False
        self.audio_thread = None
        self.configs = []
        self.trigger_count = 0
        self.last_visual_feedback = 0
        self.config_file = Path.home() / ".sonictrigger_config.json"
        self.setup_styles()
        self.create_ui()
        self.load_configurations()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        colors = {
            'bg': '#1a1a1a',
            'panel': '#252526',
            'panel_active': '#2d3748',
            'panel_disabled': '#1e1e1e',
            'border': '#3e3e42',
            'accent': '#6c5ce7',
            'accent2': '#00cec9',
            'success': '#00b894',
            'warning': '#fdcb6e',
            'error': '#ff7675',
            'text': '#ffffff',
            'text_dim': '#a0a0a0',
            'slider_trough': '#333333',
            'slider_slider': '#6c5ce7'
        }
        
        style.configure('TFrame', background=colors['bg'])
        style.configure('Main.TFrame', background=colors['bg'])
        style.configure('Header.TFrame', background=colors['bg'])
        style.configure('Stats.TFrame', background=colors['panel'], relief='flat')
        style.configure('Config.TFrame', background=colors['panel'],
                      relief='flat', borderwidth=1, bordercolor=colors['border'])
        style.configure('ConfigActive.TFrame', background=colors['panel_active'],
                      relief='flat', borderwidth=1, bordercolor=colors['accent'])
        style.configure('ConfigDisabled.TFrame', background=colors['panel_disabled'],
                      relief='flat', borderwidth=1, bordercolor=colors['border'])
        style.configure('ConfigHeader.TFrame', background=colors['panel'], relief='flat')
        style.configure('ConfigContent.TFrame', background=colors['panel'], relief='flat')
        style.configure('ConfigSection.TFrame', background=colors['panel'], relief='flat')
        style.configure('TLabel', background=colors['bg'], foreground=colors['text'], font=('Segoe UI', 10))
        style.configure('Header.TLabel', background=colors['bg'], foreground=colors['accent'],
                      font=('Segoe UI', 20, 'bold'))
        style.configure('Subheader.TLabel', background=colors['bg'], foreground=colors['text_dim'],
                      font=('Segoe UI', 11))
        style.configure('ConfigSectionLabel.TLabel', background=colors['panel'],
                      foreground=colors['accent2'], font=('Segoe UI', 9, 'bold'))
        style.configure('ConfigSliderLabel.TLabel', background=colors['panel'],
                      foreground=colors['text'], font=('Segoe UI', 9))
        style.configure('ConfigSliderValue.TLabel', background=colors['panel'],
                      foreground=colors['accent'], font=('Segoe UI', 9, 'bold'))
        style.configure('ConfigHint.TLabel', background=colors['panel'],
                      foreground=colors['text_dim'], font=('Segoe UI', 8))
        style.configure('ConfigStatusActive.TLabel', background=colors['panel'],
                      foreground=colors['success'], font=('Segoe UI', 9))
        style.configure('ConfigStatusInactive.TLabel', background=colors['panel'],
                      foreground=colors['warning'], font=('Segoe UI', 9))
        style.configure('ConfigStatusDisabled.TLabel', background=colors['panel'],
                      foreground=colors['text_dim'], font=('Segoe UI', 9))
        style.configure('ConfigStatusError.TLabel', background=colors['panel'],
                      foreground=colors['error'], font=('Segoe UI', 9))
        style.configure('ConfigStatusBusy.TLabel', background=colors['panel'],
                      foreground=colors['accent2'], font=('Segoe UI', 9))
        style.configure('StatsValue.TLabel', background=colors['panel'],
                      foreground=colors['accent'], font=('Segoe UI', 14, 'bold'))
        style.configure('StatsLabel.TLabel', background=colors['panel'],
                      foreground=colors['text_dim'], font=('Segoe UI', 9))
        style.configure('Status.TLabel', background=colors['bg'],
                      foreground=colors['warning'], font=('Segoe UI', 11, 'bold'))
        style.configure('StatusActive.TLabel', background=colors['bg'],
                      foreground=colors['success'], font=('Segoe UI', 11, 'bold'))
        
        style.configure('MainButton.TButton', background=colors['accent'], foreground=colors['text'],
                      font=('Segoe UI', 10, 'bold'), borderwidth=0)
        style.map('MainButton.TButton', background=[('active', '#5a4fcf')])
        style.configure('StopButton.TButton', background=colors['error'], foreground=colors['text'],
                      font=('Segoe UI', 10, 'bold'), borderwidth=0)
        style.map('StopButton.TButton', background=[('active', '#e06666')])
        style.configure('ConfigButton.TButton', background=colors['border'], foreground=colors['text'],
                      font=('Segoe UI', 9), borderwidth=0, padding=(4, 2))
        style.map('ConfigButton.TButton', background=[('active', '#4a4a4f')])
        style.configure('ConfigRecord.TButton', background=colors['accent2'], foreground=colors['text'],
                      font=('Segoe UI', 9, 'bold'), borderwidth=0, padding=(4, 2))
        style.map('ConfigRecord.TButton', background=[('active', '#00a7a3')])
        style.configure('ConfigAction.TButton', background=colors['accent'], foreground=colors['text'],
                      font=('Segoe UI', 9, 'bold'), borderwidth=0, padding=(4, 2))
        style.map('ConfigAction.TButton', background=[('active', '#5a4fcf')])
        style.configure('ConfigEdit.TButton', background=colors['text_dim'], foreground=colors['bg'],
                      font=('Segoe UI', 9, 'bold'), borderwidth=0, padding=(4, 2))
        style.map('ConfigEdit.TButton', background=[('active', '#888888')])
        style.configure('ConfigDelete.TButton', background=colors['error'], foreground=colors['text'],
                      font=('Segoe UI', 9, 'bold'), borderwidth=0, padding=(4, 2))
        style.map('ConfigDelete.TButton', background=[('active', '#e06666')])
        style.configure('ConfigToggle.TCheckbutton', background=colors['panel'],
                      foreground=colors['text'], font=('Segoe UI', 9))
        style.configure('ConfigName.TEntry', fieldbackground=colors['panel'],
                      foreground=colors['accent'], insertcolor=colors['text'],
                      font=('Segoe UI', 10, 'bold'), borderwidth=0)
        style.configure('ConfigPath.TEntry', fieldbackground=colors['panel'],
                      foreground=colors['text_dim'], insertcolor=colors['text'],
                      font=('Segoe UI', 9), borderwidth=0)
        
        style.configure('ConfigSlider.Horizontal.TScale', background=colors['panel'],
                      troughcolor=colors['slider_trough'], borderwidth=0)
        style.map('ConfigSlider.Horizontal.TScale',
                sliderbackground=[('active', colors['slider_slider'])])
        
        style.configure('Vertical.TScrollbar', background=colors['panel'],
                      troughcolor=colors['bg'], borderwidth=0,
                      arrowcolor=colors['text_dim'], darkcolor=colors['bg'],
                      lightcolor=colors['bg'])
    
    def create_ui(self):
        header = ttk.Frame(self.root, style='Header.TFrame')
        header.pack(fill='x', padx=20, pady=15)
        title = ttk.Label(header, text="SonicTrigger", style='Header.TLabel')
        title.pack(anchor='w')
        subtitle = ttk.Label(header, text="Запускайте приложения по звуковому сигналу • Несколько профилей",
                           style='Subheader.TLabel')
        subtitle.pack(anchor='w', pady=(2, 0))
        
        control_frame = ttk.Frame(self.root, style='Main.TFrame')
        control_frame.pack(fill='x', padx=20, pady=(0, 15))
        self.status_label = ttk.Label(control_frame, text="● Статус: Ожидание", style='Status.TLabel')
        self.status_label.pack(side='left', padx=(0, 20))
        
        btn_frame = ttk.Frame(control_frame, style='Main.TFrame')
        btn_frame.pack(side='right')
        self.listen_btn = ttk.Button(btn_frame, text="▶ Начать прослушивание",
                                   style='MainButton.TButton', command=self.toggle_listening)
        self.listen_btn.pack(side='left', padx=(0, 10))
        ttk.Button(btn_frame, text="✚ Добавить триггер",
                 style='ConfigAction.TButton', command=self.add_config).pack(side='left', padx=(0, 10))
        ttk.Button(btn_frame, text="💾 Сохранить",
                 style='ConfigButton.TButton', command=self.save_configurations).pack(side='left', padx=(0, 5))
        ttk.Button(btn_frame, text="📂 Загрузить",
                 style='ConfigButton.TButton', command=self.load_configurations).pack(side='left')
        
        canvas_frame = ttk.Frame(self.root, style='Main.TFrame')
        canvas_frame.pack(fill='both', expand=True, padx=20, pady=(0, 15))
        canvas = tk.Canvas(canvas_frame, bg='#1a1a1a', highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas, style='Main.TFrame')
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        
        stats_frame = ttk.Frame(self.root, style='Stats.TFrame')
        stats_frame.pack(fill='x', padx=20, pady=(0, 15))
        ttk.Label(stats_frame, text="Всего срабатываний:", style='StatsLabel.TLabel').pack(side='left', padx=(15, 5))
        self.triggers_label = ttk.Label(stats_frame, text="0", style='StatsValue.TLabel')
        self.triggers_label.pack(side='left', padx=(0, 20))
        ttk.Label(stats_frame, text="Активных профилей:", style='StatsLabel.TLabel').pack(side='left', padx=(15, 5))
        self.active_label = ttk.Label(stats_frame, text="0", style='StatsValue.TLabel')
        self.active_label.pack(side='left', padx=(0, 20))
        ttk.Label(stats_frame, text="Статус:", style='StatsLabel.TLabel').pack(side='left', padx=(15, 5))
        self.stats_status = ttk.Label(stats_frame, text="Ожидание", style='StatsValue.TLabel')
        self.stats_status.pack(side='left')
        
        hint_frame = ttk.Frame(self.root, style='Main.TFrame')
        hint_frame.pack(fill='x', padx=20, pady=(0, 10))
        hint = ttk.Label(hint_frame,
                       text="💡 Совет: Используйте резкие звуки (хлопки, щелчки) для лучшего распознавания. Настройте чувствительность под уровень шума в помещении.",
                       style='Subheader.TLabel', foreground='#777777', font=('Segoe UI', 9))
        hint.pack(anchor='w')
    
    def add_config(self, config_data=None):
        panel = ConfigPanel(self.scrollable_frame, self)
        if config_data:
            panel.from_dict(config_data)
        panel.pack(fill='x', pady=8, padx=5)
        self.configs.append(panel)
        self.update_stats()
        return panel
    
    def remove_config(self, panel):
        panel.pack_forget()
        self.configs.remove(panel)
        self.update_stats()
    
    def update_stats(self):
        active = sum(1 for c in self.configs if c.enabled_var.get() and c.is_trained)
        self.active_label.config(text=str(active))
        self.triggers_label.config(text=str(self.trigger_count))
        self.stats_status.config(text="Прослушивание" if self.is_listening else "Ожидание")
    
    def toggle_listening(self):
        if not self.is_listening:
            active = [c for c in self.configs if c.enabled_var.get() and c.is_trained]
            if not active:
                messagebox.showwarning("Нет активных триггеров",
                                     "Добавьте хотя бы один обученный и включённый профиль!")
                return
            self.is_listening = True
            self.status_label.config(text="● Статус: Прослушивание", style='StatusActive.TLabel')
            self.listen_btn.config(text="⏹ Остановить", style='StopButton.TButton')
            self.audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
            self.audio_thread.start()
        else:
            self.is_listening = False
            self.status_label.config(text="● Статус: Остановка...", style='Status.TLabel')
            self.listen_btn.config(text="▶ Начать прослушивание", style='MainButton.TButton')
    
    def audio_loop(self):
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=self.FORMAT,
                          channels=self.CHANNELS,
                          rate=self.RATE,
                          input=True,
                          frames_per_buffer=self.CHUNK)
            buffer = []
            self.status_label.config(text="● Статус: Прослушивание", style='StatusActive.TLabel')
            self.update_stats()
            while self.is_listening:
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    buffer.append(data)
                    if len(buffer) > self.BUFFER_SIZE:
                        buffer.pop(0)
                    if len(buffer) == self.BUFFER_SIZE:
                        audio_data = b''.join(buffer)
                        self.process_audio(audio_data)
                except Exception as e:
                    print(f"Audio error: {e}")
                if not self.is_listening:
                    break
                time.sleep(0.01)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка аудио",
                                                          f"Не удалось получить доступ к микрофону:\n{str(e)}\nУбедитесь, что разрешения на микрофон включены."))
        finally:
            try:
                stream.stop_stream()
                stream.close()
                p.terminate()
            except:
                pass
            if not self.is_listening:
                self.root.after(0, lambda: self.status_label.config(
                    text="● Статус: Ожидание", style='Status.TLabel'))
                self.root.after(0, lambda: self.listen_btn.config(
                    text="▶ Начать прослушивание", style='MainButton.TButton'))
                self.root.after(0, self.update_stats)
    
    def process_audio(self, audio_data):
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        if len(audio_array) == 0:
            return
        rms = np.sqrt(np.mean(audio_array**2))
        normalized_volume = rms / 32768.0
        current_time = time.time()
        
        for config in self.configs:
            if not (config.enabled_var.get() and config.is_trained):
                continue
            if normalized_volume < config.data['min_volume']:
                continue
            if current_time - config.last_trigger < config.cooldown:
                continue
            distance = self.compare_audio(audio_array, config.sound_model)
            if distance < config.data['threshold']:
                config.last_trigger = current_time
                self.trigger_count += 1
                self.root.after(0, lambda c=config, d=distance: self.trigger_action(c, d))
                self.root.after(0, self.visual_feedback)
                time.sleep(0.15)
    
    def compare_audio(self, audio_array, model):
        try:
            y = audio_array / 32768.0
            if len(y) < self.RATE * 0.4:
                return float('inf')
            y = librosa.util.normalize(y)
            mfcc = librosa.feature.mfcc(
                y=y, sr=self.RATE, n_mfcc=13, n_fft=512, hop_length=256
            )
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            features = np.vstack([mfcc, mfcc_delta, mfcc_delta2]).T
            distance, _ = fastdtw(features, model['mfcc'], dist=euclidean)
            return distance / model['frames']
        except Exception as e:
            print(f"Comparison error: {e}")
            return float('inf')
    
    def trigger_action(self, config, distance):
        exe_path = config.data['exe_path'].strip()
        if not exe_path:
            self.root.after(0, lambda: messagebox.showwarning("Нет приложения",
                                                          f"В профиле '{config.data['name']}' не указано приложение!"))
            return
        if not os.path.exists(exe_path):
            self.root.after(0, lambda: messagebox.showerror("Файл не найден",
                                                          f"Приложение не найдено:\n{exe_path}"))
            return
        
        original_bg = config.status_label.cget('style')
        config.status_label.configure(text=f"⬤ Сработало! ({distance:.1f})",
                                    style='ConfigStatusActive.TLabel')
        try:
            subprocess.Popen([exe_path], shell=True, cwd=os.path.dirname(exe_path))
            self.root.after(2000, lambda: config.status_label.configure(
                text=f"⬤ Готов ({config.sound_model['duration']:.1f}с)",
                style='ConfigStatusActive.TLabel'))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка запуска",
                                                          f"Не удалось запустить {os.path.basename(exe_path)}:\n{str(e)}"))
            self.root.after(2000, lambda: config.status_label.configure(
                text="⬤ Ошибка запуска", style='ConfigStatusError.TLabel'))
        self.update_stats()
    
    def visual_feedback(self):
        current_time = time.time()
        if current_time - self.last_visual_feedback < 0.3:
            return
        self.last_visual_feedback = current_time
        original_bg = self.root.cget('bg')
        self.root.configure(bg='#ff5252')
        self.status_label.config(text="● СИГНАЛ ОБНАРУЖЕН!", style='StatusActive.TLabel')
        def restore():
            self.root.configure(bg=original_bg)
            if self.is_listening:
                self.status_label.config(text="● Статус: Прослушивание", style='StatusActive.TLabel')
        self.root.after(150, restore)
    
    def record_sound_for_config(self, config):
        def record():
            try:
                p = pyaudio.PyAudio()
                stream = p.open(format=self.FORMAT,
                              channels=self.CHANNELS,
                              rate=self.RATE,
                              input=True,
                              frames_per_buffer=self.CHUNK)
                for i in range(3, 0, -1):
                    self.root.after(0, lambda i=i: messagebox.showinfo("Запись",
                                                                  f"Запись начнётся через {i}...\nИзготовьте звук-триггер!"))
                    time.sleep(1)
                self.root.after(0, lambda: self.status_label.config(
                    text="● Запись звука...", style='Status.TLabel'))
                frames = []
                for _ in range(0, int(self.RATE / self.CHUNK * 2.0)):
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    frames.append(data)
                stream.stop_stream()
                stream.close()
                p.terminate()
                
                filename = filedialog.asksaveasfilename(
                    defaultextension=".wav",
                    filetypes=[("WAV файлы", "*.wav")],
                    initialfile=f"trigger_{config.data['name'].replace(' ', '_')}.wav",
                    title="Сохранить звук-триггер"
                )
                if filename:
                    wf = wave.open(filename, 'wb')
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(p.get_sample_size(self.FORMAT))
                    wf.setframerate(self.RATE)
                    wf.writeframes(b''.join(frames))
                    wf.close()
                    
                    self.root.after(0, lambda: config.sound_path_var.set(filename))
                    self.root.after(0, lambda: config.data.update({'sound_path': filename}))
                    self.root.after(0, config.train_model)
                    self.root.after(0, lambda: self.status_label.config(
                        text="● Статус: Ожидание", style='Status.TLabel' if not self.is_listening else 'StatusActive.TLabel'))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Ошибка записи", str(e)))
                self.root.after(0, lambda: self.status_label.config(
                    text="● Статус: Ожидание", style='Status.TLabel'))
            self.is_listening = False
        threading.Thread(target=record, daemon=True).start()
    
    def test_single_config(self, config):
        def test():
            try:
                p = pyaudio.PyAudio()
                stream = p.open(format=self.FORMAT,
                              channels=self.CHANNELS,
                              rate=self.RATE,
                              input=True,
                              frames_per_buffer=self.CHUNK)
                self.root.after(0, lambda: messagebox.showinfo("Режим теста",
                                                            "Изготовьте звук-триггер сейчас (2 секунды)..."))
                time.sleep(0.5)
                frames = []
                for _ in range(0, int(self.RATE / self.CHUNK * 2.0)):
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    frames.append(data)
                stream.stop_stream()
                stream.close()
                p.terminate()
                
                audio_data = b''.join(frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                distance = self.compare_audio(audio_array, config.sound_model)
                
                if distance < config.data['threshold']:
                    result = f"✅ СРАБОТАЛО! (дистанция: {distance:.2f})"
                    color = '#00b894'
                else:
                    result = f"❌ Не сработало (дистанция: {distance:.2f} > порог {config.data['threshold']:.1f})"
                    color = '#ff7675'
                self.root.after(0, lambda: messagebox.showinfo("Результат теста",
                                                            f"Профиль: {config.data['name']}\n{result}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Ошибка теста", str(e)))
        threading.Thread(target=test, daemon=True).start()
    
    def save_configurations(self):
        configs = [panel.to_dict() for panel in self.configs]
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(configs, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Сохранено", f"Конфигурация сохранена в:\n{self.config_file}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить конфигурацию:\n{str(e)}")
    
    def load_configurations(self):
        if not self.config_file.exists():
            default_config = [{
                'id': str(uuid.uuid4()),
                'name': 'Хлопок',
                'sound_path': '',
                'exe_path': '',
                'threshold': 2.5,
                'min_volume': 0.008,
                'enabled': True
            }]
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
            configs = default_config
        else:
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    configs = json.load(f)
            except Exception as e:
                messagebox.showwarning("Предупреждение",
                                     f"Не удалось загрузить конфигурацию:\n{str(e)}\nИспользуется конфигурация по умолчанию.")
                configs = []
        
        for panel in self.configs[:]:
            self.remove_config(panel)
        
        if configs:
            for config_data in configs:
                self.add_config(config_data)
        else:
            self.add_config()
        self.update_stats()
        messagebox.showinfo("Загружено", f"Конфигурация загружена из:\n{self.config_file}")
    
    def on_close(self):
        self.is_listening = False
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        self.root.destroy()

def check_dependencies():
    missing = []
    try:
        import pyaudio
    except ImportError:
        missing.append("pyaudio")
    try:
        import librosa
    except ImportError:
        missing.append("librosa")
    try:
        from fastdtw import fastdtw
    except ImportError:
        missing.append("fastdtw")
    if missing:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Отсутствуют зависимости",
                           "Требуемые пакеты не установлены:\n" +
                           "\n".join(f" • {pkg}" for pkg in missing) +
                           "\nУстановите командой:\npip install " + " ".join(missing))
        return False
    return True

def main():
    if not check_dependencies():
        return
    root = tk.Tk()
    app = SoundTriggerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()