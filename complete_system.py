"""
=====================================================
البرنامج الكامل المدمج - نظام تشخيص أمراض القلب
Complete Integrated System - Heart Disease Diagnosis
=====================================================

هذا البرنامج يجمع كل شيء في مكان واحد:
1. تدريب النموذج
2. قراءة البيانات من الأردوينو
3. التنبؤ بحالة القلب
4. عرض النتائج
5. حفظ النتائج
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from scipy import signal
import pickle
import serial
import time
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================
# الإعدادات
# ============================================

SERIAL_PORT = 'COM3'      # غيّر هذا حسب منفذك
BAUD_RATE = 115200
TIMEOUT = 1
MODEL_PATH = 'heart_disease_model.pkl'
RESULTS_FILE = 'ecg_results.txt'

# ============================================
# القسم 1: تدريب النموذج
# ============================================

def generate_normal_heartbeats(num_beats=300, fs=360):
    """توليد نبضات قلب طبيعية"""
    heartbeats = []
    beat_duration = int(fs * 0.8)
    
    for _ in range(num_beats):
        t = np.linspace(0, 0.8, beat_duration)
        
        p_wave = 0.15 * np.exp(-((t - 0.1) ** 2) / 0.01)
        q_wave = -0.2 * np.exp(-((t - 0.3) ** 2) / 0.002)
        r_wave = 1.0 * np.exp(-((t - 0.32) ** 2) / 0.002)
        s_wave = -0.3 * np.exp(-((t - 0.34) ** 2) / 0.002)
        t_wave = 0.3 * np.exp(-((t - 0.5) ** 2) / 0.015)
        st_segment = np.zeros_like(t)
        
        ecg = p_wave + q_wave + r_wave + s_wave + t_wave + st_segment
        noise = np.random.normal(0, 0.05, len(ecg))
        ecg = ecg + noise
        
        heartbeats.append(ecg)
    
    return np.array(heartbeats)


def generate_ischemia_heartbeats(num_beats=300, fs=360):
    """توليد نبضات قلب مع نقص تروية"""
    heartbeats = []
    beat_duration = int(fs * 0.8)
    
    for _ in range(num_beats):
        t = np.linspace(0, 0.8, beat_duration)
        
        p_wave = 0.15 * np.exp(-((t - 0.1) ** 2) / 0.01)
        q_wave = -0.15 * np.exp(-((t - 0.3) ** 2) / 0.002)
        r_wave = 0.8 * np.exp(-((t - 0.32) ** 2) / 0.002)
        s_wave = -0.25 * np.exp(-((t - 0.34) ** 2) / 0.002)
        t_wave = -0.3 * np.exp(-((t - 0.5) ** 2) / 0.015)
        st_segment = np.where((t > 0.34) & (t < 0.5), -0.25, 0)
        
        ecg = p_wave + q_wave + r_wave + s_wave + t_wave + st_segment
        noise = np.random.normal(0, 0.05, len(ecg))
        ecg = ecg + noise
        
        heartbeats.append(ecg)
    
    return np.array(heartbeats)


def extract_features(heartbeat):
    """استخلاص 10 ميزات من نبضة القلب"""
    features = []
    
    features.append(np.max(heartbeat))
    features.append(np.min(heartbeat))
    features.append(np.mean(heartbeat))
    features.append(np.std(heartbeat))
    features.append(np.sum(heartbeat ** 2))
    features.append(np.mean((heartbeat - np.mean(heartbeat)) ** 3) / (np.std(heartbeat) ** 3))
    features.append(np.mean((heartbeat - np.mean(heartbeat)) ** 4) / (np.std(heartbeat) ** 4))
    
    diff = np.diff(heartbeat)
    features.append(np.mean(np.abs(diff)))
    
    mid_point = len(heartbeat) // 2
    st_region = heartbeat[int(mid_point * 0.85):int(mid_point * 1.15)]
    features.append(np.mean(st_region))
    
    zero_crossings = np.sum(np.abs(np.diff(np.sign(heartbeat)))) / 2
    features.append(zero_crossings)
    
    return np.array(features)


def train_model():
    """تدريب النموذج وحفظه"""
    print("\n" + "="*70)
    print("🤖 تدريب نموذج الذكاء الاصطناعي")
    print("="*70)
    
    print("\n📊 جاري توليد البيانات التدريبية...")
    normal_beats = generate_normal_heartbeats(num_beats=300)
    ischemia_beats = generate_ischemia_heartbeats(num_beats=300)
    
    print(f"✅ تم توليد {len(normal_beats)} نبضة طبيعية")
    print(f"✅ تم توليد {len(ischemia_beats)} نبضة مريضة")
    
    print("\n🔍 جاري استخلاص الميزات...")
    normal_features = np.array([extract_features(beat) for beat in normal_beats])
    ischemia_features = np.array([extract_features(beat) for beat in ischemia_beats])
    
    X = np.vstack([normal_features, ischemia_features])
    y = np.hstack([np.zeros(len(normal_features)), np.ones(len(ischemia_features))])
    
    print("\n📈 تقسيم البيانات...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ بيانات التدريب: {len(X_train)} نبضة")
    print(f"✅ بيانات الاختبار: {len(X_test)} نبضة")
    
    print("\n🤖 جاري تدريب النموذج...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("✅ تم تدريب النموذج بنجاح!")
    
    # التقييم
    print("\n📊 تقييم النموذج...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n{'='*70}")
    print(f"📈 نتائج الأداء:")
    print(f"{'='*70}")
    print(f"✅ الدقة (Accuracy):        {accuracy:.2%}")
    print(f"✅ الدقة الموجبة (Precision): {precision:.2%}")
    print(f"✅ الاستدعاء (Recall):      {recall:.2%}")
    print(f"✅ درجة F1 (F1-Score):      {f1:.2%}")
    print(f"{'='*70}\n")
    
    # حفظ النموذج
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ تم حفظ النموذج في: {MODEL_PATH}")
    
    return model


# ============================================
# القسم 2: قراءة البيانات من الأردوينو
# ============================================

def load_model():
    """تحميل النموذج المدرب"""
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ تم تحميل النموذج من: {MODEL_PATH}")
        return model
    except FileNotFoundError:
        print(f"❌ خطأ: لم يتم العثور على النموذج!")
        print(f"   يرجى تشغيل train_model() أولاً")
        return None


def connect_to_arduino(port=SERIAL_PORT, baudrate=BAUD_RATE):
    """الاتصال بالأردوينو"""
    try:
        ser = serial.Serial(port, baudrate, timeout=TIMEOUT)
        time.sleep(2)
        print(f"✅ تم الاتصال بالأردوينو على المنفذ: {port}")
        return ser
    except serial.SerialException as e:
        print(f"❌ خطأ في الاتصال: {e}")
        print(f"   تأكد من:")
        print(f"   1. توصيل الأردوينو بـ USB")
        print(f"   2. اختيار المنفذ الصحيح (حالياً: {port})")
        return None


def read_heartbeat_data(ser):
    """قراءة بيانات نبضة قلب من الأردوينو"""
    try:
        while True:
            line = ser.readline().decode('utf-8').strip()
            
            if line.startswith('HEARTBEAT:'):
                features_str = line.replace('HEARTBEAT:', '')
                features = [float(x) for x in features_str.split(',')]
                
                if len(features) == 10:
                    return np.array(features)
                    
            elif line:
                print(f"📡 {line}")
                
    except ValueError as e:
        print(f"❌ خطأ في تحليل البيانات: {e}")
        return None
    except KeyboardInterrupt:
        print("\n🛑 تم إيقاف البرنامج")
        return None


# ============================================
# القسم 3: التنبؤ والعرض
# ============================================

def predict_heart_condition(model, features):
    """التنبؤ بحالة القلب"""
    features_reshaped = features.reshape(1, -1)
    prediction = model.predict(features_reshaped)[0]
    probability = model.predict_proba(features_reshaped)[0]
    
    return prediction, probability


def display_results(features, prediction, probability):
    """عرض النتائج"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*70)
    print(f"⏰ الوقت: {timestamp}")
    print("="*70)
    
    print("\n📊 الميزات المستخلصة:")
    feature_names = [
        "القيمة العظمى",
        "القيمة الصغرى",
        "المتوسط",
        "الانحراف المعياري",
        "الطاقة",
        "الانحراف (Skewness)",
        "التفرطح (Kurtosis)",
        "معدل التغيير",
        "متوسط ST ⭐",
        "تقاطعات الصفر"
    ]
    
    for i, (name, value) in enumerate(zip(feature_names, features)):
        print(f"   {i+1:2d}. {name:20s}: {value:10.2f}")
    
    print("\n" + "="*70)
    print("🔮 نتيجة التنبؤ:")
    print("="*70)
    
    if prediction == 0:
        status = "✅ قلب طبيعي (Normal)"
        color = "🟢"
    else:
        status = "⚠️ نقص تروية (Ischemia/Atherosclerosis)"
        color = "🔴"
    
    print(f"\n{color} {status}")
    print(f"\n📈 درجة الثقة:")
    print(f"   - احتمال قلب طبيعي:      {probability[0]:.2%}")
    print(f"   - احتمال نقص تروية:     {probability[1]:.2%}")
    
    if prediction == 1:
        risk_level = probability[1]
        if risk_level > 0.9:
            print(f"\n🚨 مستوى الخطورة: **عالي جداً** ({risk_level:.2%})")
        elif risk_level > 0.7:
            print(f"\n⚠️ مستوى الخطورة: **عالي** ({risk_level:.2%})")
        else:
            print(f"\n⚠️ مستوى الخطورة: **متوسط** ({risk_level:.2%})")
    else:
        print(f"\n✅ مستوى الخطورة: **منخفض جداً**")
    
    print("\n" + "="*70 + "\n")


def save_results(features, prediction, probability):
    """حفظ النتائج"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"⏰ الوقت: {timestamp}\n")
        f.write(f"{'='*70}\n")
        
        f.write("\n📊 الميزات:\n")
        for i, value in enumerate(features):
            f.write(f"   الميزة {i+1}: {value:.2f}\n")
        
        status = "طبيعي" if prediction == 0 else "نقص تروية"
        f.write(f"\n🔮 النتيجة: {status}\n")
        f.write(f"   احتمال طبيعي: {probability[0]:.2%}\n")
        f.write(f"   احتمال مرض: {probability[1]:.2%}\n")
        f.write(f"\n")


# ============================================
# القسم 4: البرنامج الرئيسي
# ============================================

def main_menu():
    """القائمة الرئيسية"""
    print("\n" + "="*70)
    print("🏥 نظام تشخيص أمراض القلب - البرنامج الكامل المدمج")
    print("="*70)
    print("\nاختر ما تريد:")
    print("1️⃣  تدريب النموذج (المرة الأولى فقط)")
    print("2️⃣  قراءة البيانات من الأردوينو والتنبؤ")
    print("3️⃣  عرض النتائج المحفوظة")
    print("4️⃣  خروج")
    print("\n" + "="*70)


def view_results():
    """عرض النتائج المحفوظة"""
    if not os.path.exists(RESULTS_FILE):
        print(f"\n❌ لا توجد نتائج محفوظة بعد!")
        return
    
    print(f"\n📋 النتائج المحفوظة في: {RESULTS_FILE}\n")
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        print(f.read())


def run_prediction():
    """تشغيل برنامج التنبؤ"""
    model = load_model()
    if model is None:
        print("\n⚠️ يجب تدريب النموذج أولاً!")
        return
    
    ser = connect_to_arduino()
    if ser is None:
        return
    
    print("\n📡 جاري استقبال رسالة البداية من الأردوينو...")
    time.sleep(2)
    ser.reset_input_buffer()
    
    print("\n🚀 جاري قراءة البيانات... (اضغط Ctrl+C للإيقاف)\n")
    
    heartbeat_count = 0
    
    try:
        while True:
            features = read_heartbeat_data(ser)
            
            if features is not None:
                heartbeat_count += 1
                prediction, probability = predict_heart_condition(model, features)
                display_results(features, prediction, probability)
                save_results(features, prediction, probability)
                
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print(f"🛑 تم إيقاف البرنامج")
        print(f"📊 إجمالي النبضات المقروءة: {heartbeat_count}")
        print("="*70 + "\n")
    
    finally:
        if ser.is_open:
            ser.close()
            print("✅ تم إغلاق الاتصال بالأردوينو")


def main():
    """البرنامج الرئيسي"""
    while True:
        main_menu()
        choice = input("اختيارك: ").strip()
        
        if choice == '1':
            train_model()
        elif choice == '2':
            run_prediction()
        elif choice == '3':
            view_results()
        elif choice == '4':
            print("\n👋 وداعاً!")
            break
        else:
            print("\n❌ اختيار غير صحيح!")


if __name__ == '__main__':
    main()

"""
=====================================================
كيفية الاستخدام:
=====================================================

1. تثبيت المكتبات:
   pip install pyserial scikit-learn numpy matplotlib seaborn scipy

2. تشغيل البرنامج:
   python3 complete_system.py

3. الخطوات:
   - اختر 1 لتدريب النموذج (المرة الأولى فقط)
   - اختر 2 لقراءة البيانات والتنبؤ
   - اختر 3 لعرض النتائج المحفوظة
   - اختر 4 للخروج

=====================================================
"""
