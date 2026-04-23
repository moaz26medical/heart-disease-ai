import tkinter as tk
from tkinter import ttk, messagebox
import serial
import serial.tools.list_ports
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import collections
import time

# --- Global Variables and Model Loading ---
MODEL_FILE = 'atherosclerosis_ai_model.pkl'
model = None

try:
    model = joblib.load(MODEL_FILE)
    print("✅ تم تحميل نموذج الذكاء الاصطناعي بنجاح.")
except Exception as e:
    messagebox.showerror("خطأ في تحميل النموذج", f"❌ فشل تحميل نموذج الذكاء الاصطناعي: {e}\nالرجاء التأكد من وجود ملف '{MODEL_FILE}' في نفس مجلد البرنامج.")
    exit()

# --- Feature Extraction Function (Same as before) ---
def extract_features_from_signal(signal_window):
    # This is a simplified feature extraction. For real medical applications,
    # more sophisticated methods are needed to detect P, QRS, T waves accurately.
    if len(signal_window) < 500: # Ensure enough data points for meaningful analysis
        return np.zeros(3) # Return dummy features if not enough data

    signal_array = np.array(signal_window)

    # Basic statistical features
    mean_val = np.mean(signal_array)
    std_dev = np.std(signal_array)
    peak_to_peak = np.max(signal_array) - np.min(signal_array)

    # Placeholder for more advanced ECG features (e.g., ST-segment, T-wave morphology)
    # For a real project, you would implement algorithms to detect R-peaks,
    # then identify P, QRS, T waves and extract features like:
    # - ST-segment elevation/depression
    # - T-wave inversion/flattening
    # - QRS duration
    # - Heart Rate Variability (HRV) - requires more complex analysis over time

    # For this demo, we'll use simplified features that might correlate with signal changes
    # A more robust solution would involve proper ECG waveform analysis.
    feature1 = mean_val
    feature2 = std_dev
    feature3 = peak_to_peak

    return np.array([feature1, feature2, feature3]).reshape(1, -1)

# --- GUI Application Class ---
class HeartDiseaseMonitorApp:
    def __init__(self, master):
        self.master = master
        master.title("نظام تشخيص أمراض القلب بالذكاء الاصطناعي")
        master.geometry("1000x700")

        self.serial_port = None
        self.ser = None
        self.is_running = False
        self.data_buffer = collections.deque(np.zeros(500), maxlen=500) # Buffer for plotting
        self.signal_window_for_ai = collections.deque(maxlen=1000) # Larger buffer for AI analysis

        self.create_widgets()
        self.update_plot()

    def create_widgets(self):
        # --- Control Frame ---
        control_frame = ttk.LabelFrame(self.master, text="إعدادات الاتصال والتحكم", padding="10 10 10 10")
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        ttk.Label(control_frame, text="اختر منفذ الأردوينو:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.port_combobox = ttk.Combobox(control_frame, width=30)
        self.port_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.refresh_ports_button = ttk.Button(control_frame, text="تحديث المنافذ", command=self.refresh_ports)
        self.refresh_ports_button.grid(row=0, column=2, padx=5, pady=5)

        self.start_button = ttk.Button(control_frame, text="بدء المراقبة", command=self.start_monitoring, state=tk.DISABLED)
        self.start_button.grid(row=0, column=3, padx=5, pady=5)
        self.stop_button = ttk.Button(control_frame, text="إيقاف المراقبة", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=4, padx=5, pady=5)

        self.refresh_ports() # Populate ports on startup
        self.port_combobox.bind("<<ComboboxSelected>>", self.on_port_selected)

        # --- Plotting Frame ---
        plot_frame = ttk.LabelFrame(self.master, text="إشارة القلب (ECG) في الوقت الفعلي", padding="10 10 10 10")
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.line, = self.ax.plot(list(self.data_buffer))
        self.ax.set_ylim(0, 1024) # Arduino analog read range
        self.ax.set_title("إشارة ECG")
        self.ax.set_xlabel("الوقت")
        self.ax.set_ylabel("قيمة الحساس")
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- Diagnosis Frame ---
        diagnosis_frame = ttk.LabelFrame(self.master, text="نتائج التشخيص بالذكاء الاصطناعي", padding="10 10 10 10")
        diagnosis_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.diagnosis_label = ttk.Label(diagnosis_frame, text="الحالة: في انتظار البيانات...", font=("Arial", 16, "bold"), foreground="blue")
        self.diagnosis_label.pack(pady=10)

    def refresh_ports(self):
        ports = [port.device for port in serial.tools.list_ports.comports()]
        self.port_combobox['values'] = ports
        if ports:
            self.port_combobox.set(ports[0])
            self.serial_port = ports[0]
            self.start_button['state'] = tk.NORMAL
        else:
            self.port_combobox.set("لا توجد منافذ متاحة")
            self.serial_port = None
            self.start_button['state'] = tk.DISABLED

    def on_port_selected(self, event):
        self.serial_port = self.port_combobox.get()
        if self.serial_port:
            self.start_button['state'] = tk.NORMAL
        else:
            self.start_button['state'] = tk.DISABLED

    def start_monitoring(self):
        if self.serial_port and not self.is_running:
            try:
                self.ser = serial.Serial(self.serial_port, 9600, timeout=1)
                time.sleep(2) # Allow Arduino to reset
                self.is_running = True
                self.start_button['state'] = tk.DISABLED
                self.stop_button['state'] = tk.NORMAL
                self.diagnosis_label.config(text="الحالة: قراءة البيانات...", foreground="orange")
                self.read_serial_data()
            except serial.SerialException as e:
                messagebox.showerror("خطأ في الاتصال", f"❌ فشل الاتصال بمنفذ {self.serial_port}: {e}\nالرجاء التأكد من أن المنفذ صحيح وغير مستخدم من قبل برنامج آخر.")
                self.stop_monitoring()
        elif self.is_running:
            messagebox.showinfo("معلومة", "المراقبة تعمل بالفعل.")

    def stop_monitoring(self):
        self.is_running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.start_button['state'] = tk.NORMAL
        self.stop_button['state'] = tk.DISABLED
        self.diagnosis_label.config(text="الحالة: المراقبة متوقفة.", foreground="red")

    def read_serial_data(self):
        if self.is_running and self.ser and self.ser.is_open:
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line.isdigit():
                    value = int(line)
                    self.data_buffer.append(value)
                    self.signal_window_for_ai.append(value)

                    # Perform AI diagnosis every N data points (e.g., every 100 points)
                    if len(self.signal_window_for_ai) % 100 == 0 and len(self.signal_window_for_ai) >= 500:
                        self.perform_diagnosis()

                elif line == '0': # Assuming '0' means lead off from Arduino code
                    self.diagnosis_label.config(text="⚠️ الأقطاب مفصولة!", foreground="red")
                
            except serial.SerialException as e:
                messagebox.showerror("خطأ في القراءة", f"❌ خطأ في قراءة البيانات من المنفذ التسلسلي: {e}")
                self.stop_monitoring()
            except ValueError:
                pass # Ignore non-numeric data
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

            self.master.after(1, self.read_serial_data) # Read again quickly

    def perform_diagnosis(self):
        if model and len(self.signal_window_for_ai) >= 500:
            features = extract_features_from_signal(list(self.signal_window_for_ai))
            if features.shape[1] == model.n_features_in_:
                prediction = model.predict(features)
                prediction_proba = model.predict_proba(features)

                if prediction[0] == 1: # Assuming 1 is atherosclerosis
                    self.diagnosis_label.config(text=f"⚠️ اشتباه في عصيدة قلبية! (ثقة: {prediction_proba[0][1]*100:.2f}%) ", foreground="red")
                else:
                    self.diagnosis_label.config(text=f"✅ إشارة قلبية سليمة. (ثقة: {prediction_proba[0][0]*100:.2f}%) ", foreground="green")
            else:
                self.diagnosis_label.config(text="خطأ: عدد الميزات غير متطابق مع النموذج.", foreground="red")

    def update_plot(self):
        self.line.set_ydata(list(self.data_buffer))
        self.ax.set_xlim(0, len(self.data_buffer))
        self.canvas.draw_idle()
        self.master.after(50, self.update_plot) # Update plot every 50ms

# --- Main Application Run ---
if __name__ == "__main__":
    root = tk.Tk()
    app = HeartDiseaseMonitorApp(root)
    root.mainloop()
