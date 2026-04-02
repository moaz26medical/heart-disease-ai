"""
=====================================================
برنامج البايثون لقراءة بيانات الأردوينو والتنبؤ
Python Program for Arduino Data Reading and Prediction
=====================================================

هذا البرنامج يقرأ البيانات من الأردوينو (حساس AD8232)
ويستخدم النموذج المدرب للتنبؤ بحالة القلب
"""

import serial
import numpy as np
import pickle
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. إعدادات الاتصال التسلسلي
# ============================================

SERIAL_PORT = 'COM3'      # غير هذا حسب منفذ الأردوينو (COM3, COM4, إلخ)
BAUD_RATE = 115200        # نفس معدل البود في الأردوينو
TIMEOUT = 1               # مهلة زمنية بالثانية

# ============================================
# 2. تحميل النموذج المدرب
# ============================================

def load_model(model_path='heart_disease_model.pkl'):
    """
    تحميل النموذج المدرب من الملف
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ تم تحميل النموذج من: {model_path}")
        return model
    except FileNotFoundError:
        print(f"❌ خطأ: لم يتم العثور على ملف النموذج: {model_path}")
        return None

# ============================================
# 3. الاتصال بالأردوينو
# ============================================

def connect_to_arduino(port=SERIAL_PORT, baudrate=BAUD_RATE):
    """
    الاتصال بالأردوينو عبر المنفذ التسلسلي
    """
    try:
        ser = serial.Serial(port, baudrate, timeout=TIMEOUT)
        time.sleep(2)  # انتظر حتى يبدأ الأردوينو
        print(f"✅ تم الاتصال بالأردوينو على المنفذ: {port}")
        return ser
    except serial.SerialException as e:
        print(f"❌ خطأ في الاتصال: {e}")
        print(f"   تأكد من:")
        print(f"   1. توصيل الأردوينو بـ USB")
        print(f"   2. اختيار المنفذ الصحيح (حالياً: {port})")
        print(f"   3. إغلاق أي برنامج آخر يستخدم المنفذ")
        return None

# ============================================
# 4. قراءة البيانات من الأردوينو
# ============================================

def read_heartbeat_data(ser):
    """
    قراءة بيانات نبضة قلب كاملة من الأردوينو
    البيانات تأتي بصيغة: HEARTBEAT:f1,f2,f3,...,f10
    """
    try:
        while True:
            line = ser.readline().decode('utf-8').strip()
            
            if line.startswith('HEARTBEAT:'):
                # استخراج الميزات من السطر
                features_str = line.replace('HEARTBEAT:', '')
                features = [float(x) for x in features_str.split(',')]
                
                if len(features) == 10:
                    return np.array(features)
                else:
                    print(f"⚠️ تحذير: عدد الميزات غير صحيح ({len(features)} بدلاً من 10)")
                    
            elif line:
                # طباعة الرسائل الأخرى من الأردوينو
                print(f"📡 {line}")
                
    except ValueError as e:
        print(f"❌ خطأ في تحليل البيانات: {e}")
        return None
    except KeyboardInterrupt:
        print("\n🛑 تم إيقاف البرنامج من قبل المستخدم")
        return None

# ============================================
# 5. التنبؤ بحالة القلب
# ============================================

def predict_heart_condition(model, features):
    """
    استخدام النموذج للتنبؤ بحالة القلب
    
    النتيجة:
    - 0: قلب طبيعي (Normal)
    - 1: نقص تروية (Ischemia/Atherosclerosis)
    """
    # إعادة تشكيل البيانات
    features_reshaped = features.reshape(1, -1)
    
    # التنبؤ
    prediction = model.predict(features_reshaped)[0]
    probability = model.predict_proba(features_reshaped)[0]
    
    return prediction, probability

# ============================================
# 6. عرض النتائج
# ============================================

def display_results(features, prediction, probability):
    """
    عرض النتائج بشكل جميل وسهل الفهم
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "="*70)
    print(f"⏰ الوقت: {timestamp}")
    print("="*70)
    
    # عرض الميزات
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
    
    # عرض التنبؤ
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
    
    # تقييم الخطورة
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

# ============================================
# 7. حفظ النتائج
# ============================================

def save_results(features, prediction, probability, filename='ecg_results.txt'):
    """
    حفظ النتائج في ملف نصي
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"⏰ الوقت: {timestamp}\n")
        f.write(f"{'='*70}\n")
        
        # الميزات
        f.write("\n📊 الميزات:\n")
        for i, value in enumerate(features):
            f.write(f"   الميزة {i+1}: {value:.2f}\n")
        
        # النتيجة
        status = "طبيعي" if prediction == 0 else "نقص تروية"
        f.write(f"\n🔮 النتيجة: {status}\n")
        f.write(f"   احتمال طبيعي: {probability[0]:.2%}\n")
        f.write(f"   احتمال مرض: {probability[1]:.2%}\n")
        f.write(f"\n")

# ============================================
# 8. البرنامج الرئيسي
# ============================================

def main():
    """
    البرنامج الرئيسي
    """
    print("\n" + "="*70)
    print("🏥 نظام تشخيص أمراض القلب - قراءة البيانات من الأردوينو")
    print("Heart Disease Diagnosis System - Arduino Data Reader")
    print("="*70 + "\n")
    
    # تحميل النموذج
    model = load_model()
    if model is None:
        return
    
    # الاتصال بالأردوينو
    ser = connect_to_arduino()
    if ser is None:
        return
    
    # قراءة رسالة البداية من الأردوينو
    print("\n📡 جاري استقبال رسالة البداية من الأردوينو...")
    time.sleep(2)
    
    # تنظيف المخزن المؤقت
    ser.reset_input_buffer()
    
    print("\n🚀 جاري قراءة البيانات... (اضغط Ctrl+C للإيقاف)\n")
    
    heartbeat_count = 0
    
    try:
        while True:
            # قراءة بيانات نبضة قلب
            features = read_heartbeat_data(ser)
            
            if features is not None:
                heartbeat_count += 1
                
                # التنبؤ
                prediction, probability = predict_heart_condition(model, features)
                
                # عرض النتائج
                display_results(features, prediction, probability)
                
                # حفظ النتائج
                save_results(features, prediction, probability)
                
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print(f"🛑 تم إيقاف البرنامج")
        print(f"📊 إجمالي النبضات المقروءة: {heartbeat_count}")
        print("="*70 + "\n")
    
    finally:
        # إغلاق الاتصال
        if ser.is_open:
            ser.close()
            print("✅ تم إغلاق الاتصال بالأردوينو")

# ============================================
# 9. نقطة البداية
# ============================================

if __name__ == '__main__':
    main()

"""
=====================================================
كيفية الاستخدام:
=====================================================

1. تثبيت المكتبات المطلوبة:
   pip install pyserial scikit-learn numpy

2. تأكد من:
   - توصيل الأردوينو بـ USB
   - رفع كود الأردوينو على اللوحة
   - اختيار المنفذ الصحيح (COM3, COM4, إلخ)

3. تشغيل البرنامج:
   python3 python_arduino_integration.py

4. النتائج:
   - ستظهر على الشاشة مباشرة
   - ستُحفظ في ملف: ecg_results.txt

=====================================================
ملاحظات:
=====================================================

- غير SERIAL_PORT حسب منفذ الأردوينو لديك
- تأكد من أن معدل البود = 115200
- النموذج يتنبأ بـ:
  * 0 = قلب طبيعي
  * 1 = نقص تروية (عصيدة)
- احفظ النتائج لمراجعتها لاحقاً

=====================================================
"""
