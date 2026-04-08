import sys
import numpy as np
import tflite_runtime.interpreter as tflite
import board, busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
from luma.core.interface.serial import i2c
from luma.oled.device import sh1106
from PIL import Image, ImageDraw, ImageFont

interpreter = tflite.Interpreter(
    model_path="/home/mohammed/Desktop/ecg_cnn_model.tflite"
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
classes = ["Normal", "Ischemia", "Undiagnosed"]
window_size = 200
threshold = 0.6

serial = i2c(port=1, address=0x3C)
device = sh1106(serial, width=128, height=64)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)


def show_on_oled(lines):
    image = Image.new("1", (device.width, device.height), 0)
    draw = ImageDraw.Draw(image)
    y = 0
    for line in lines:
        draw.text((0, y), line, font=font, fill=255)
        y += 20
    device.display(image)


i2c_ads = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c_ads)
leadII_channel = AnalogIn(ads, ADS.P0)
leadIII_channel = AnalogIn(ads, ADS.P1)

app = QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(title="ECG Leads")
win.show()
win.resize(1000, 800)
plot1 = win.addPlot(title="Lead II")
curve_leadII = plot1.plot(pen="g")
win.nextRow()
plot2 = win.addPlot(title="Lead III")
curve_leadIII = plot2.plot(pen="b")
win.nextRow()
plot3 = win.addPlot(title="aVF (Computed)")
curve_avf = plot3.plot(pen="r")

fs = 250
max_points = 500
time_data = np.linspace(-max_points / fs, 0, max_points)
data_leadII = np.zeros(max_points)
data_leadIII = np.zeros(max_points)
data_avf = np.zeros(max_points)
buffer_leadII, buffer_leadIII, buffer_avf = [], [], []

counters = {
    "Lead II": {"Normal": 0, "Ischemia": 0, "Undiagnosed": 0},
    "Lead III": {"Normal": 0, "Ischemia": 0, "Undiagnosed": 0},
    "aVF": {"Normal": 0, "Ischemia": 0, "Undiagnosed": 0},
}


def update():
    global data_leadII, data_leadIII, data_avf
    vII = leadII_channel.voltage
    vIII = leadIII_channel.voltage
    vAVF = (vII + vIII) / 2
    data_leadII = np.roll(data_leadII, -1)
    data_leadII[-1] = vII
    data_leadIII = np.roll(data_leadIII, -1)
    data_leadIII[-1] = vIII
    data_avf = np.roll(data_avf, -1)
    data_avf[-1] = vAVF
    curve_leadII.setData(time_data, data_leadII)
    curve_leadIII.setData(time_data, data_leadIII)
    curve_avf.setData(time_data, data_avf)
    buffer_leadII.append(vII)
    buffer_leadIII.append(vIII)
    buffer_avf.append(vAVF)
    if len(buffer_leadII) >= window_size:
        for lead_name, buffer in zip(
            ["Lead II", "Lead III", "aVF"], [buffer_leadII, buffer_leadIII, buffer_avf]
        ):
            segment = np.array(buffer[-window_size:])
            segment = (segment - np.mean(segment)) / (np.std(segment) + 1e-8)
            data = np.array(segment, dtype=input_details[0]["dtype"]).reshape(
                input_details[0]["shape"]
            )
            interpreter.set_tensor(input_details[0]["index"], data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]["index"])[0]
            max_prob_index = np.argmax(output_data)
            max_prob = output_data[max_prob_index]
            if max_prob >= threshold:
                result = classes[max_prob_index]
            else:
                result = "Undiagnosed"
            counters[lead_name][result] += 1
        print("📊 Counters:")
        for lead, counts in counters.items():
            print(f"{lead}: {counts}")
        c = counters["Lead II"]
        lines = [
            f"Normal: {c['Normal']}",
            f"Ischemia: {c['Ischemia']}",
            f"Undiag: {c['Undiagnosed']}",
        ]
        show_on_oled(lines)


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(int(1000 / fs))
sys.exit(app.exec_())
