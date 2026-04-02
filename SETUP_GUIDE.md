# 📋 دليل الخطوات الكامل - ترتيب العمل من البداية إلى النهاية

## 🎯 الخطوات الرئيسية:

```
الخطوة 1: تدريب النموذج (Python)
    ↓
الخطوة 2: رفع كود الأردوينو
    ↓
الخطوة 3: قراءة البيانات من الأردوينو (Python)
    ↓
الخطوة 4: التنبؤ بحالة القلب
```

---

## 📁 هيكل الملفات:

```
مشروعك/
│
├── 📂 AI_Model/
│   ├── train_model.py              ← كود تدريب الذكاء الاصطناعي
│   └── heart_disease_model.pkl     ← النموذج المدرب (يُنشأ تلقائياً)
│
├── 📂 Arduino/
│   ├── arduino_ecg_sensor.ino      ← كود الأردوينو
│   └── ARDUINO_SETUP.md            ← دليل التوصيل
│
├── 📂 Python_Programs/
│   ├── python_arduino_integration.py  ← برنامج البايثون الرئيسي
│   └── ecg_results.txt             ← النتائج المحفوظة
│
└── 📂 Website/
    ├── index.html                  ← الموقع الإلكتروني
    └── images/
        └── model_results.png       ← الرسوم البيانية
```

---

## 🚀 الخطوة 1: تدريب النموذج (Python)

### ما هو هذا الكود؟
كود يقرأ بيانات ECG ويدرب نموذج Random Forest على التمييز بين:
- ✅ قلب طبيعي (Normal)
- ⚠️ قلب مريض (نقص تروية)

### الملف:
📄 **train_model.py**

### الخطوات:
```bash
# 1. انسخ الملف إلى مجلد على جهازك
cp train_model.py ~/my_project/AI_Model/

# 2. تثبيت المكتبات المطلوبة
pip install scikit-learn numpy matplotlib seaborn scipy

# 3. تشغيل الكود
cd ~/my_project/AI_Model/
python3 train_model.py
```

### ماذا سيحدث؟
```
✅ توليد 600 نبضة قلب (300 طبيعية + 300 مريضة)
✅ استخلاص 10 ميزات من كل نبضة
✅ تقسيم البيانات: 80% تدريب، 20% اختبار
✅ تدريب نموذج Random Forest
✅ حفظ النموذج في: heart_disease_model.pkl
✅ رسم الرسوم البيانية
```

### الملف الناتج:
```
✅ heart_disease_model.pkl  ← النموذج المدرب (مهم جداً!)
✅ model_results.png        ← الرسوم البيانية
```

**ملاحظة:** هذا الكود يُشغّل مرة واحدة فقط لتدريب النموذج!

---

## 🔧 الخطوة 2: رفع كود الأردوينو

### ما هو هذا الكود؟
كود يقرأ إشارة ECG من حساس AD8232 ويستخلص الميزات ويرسلها إلى الكمبيوتر

### الملف:
📄 **arduino_ecg_sensor.ino**

### الخطوات:
```
1. افتح Arduino IDE
2. انسخ محتوى arduino_ecg_sensor.ino
3. اختر اللوحة: Tools → Board → Arduino Uno
4. اختر المنفذ: Tools → Port → COM3 (أو المنفذ لديك)
5. اضغط Upload (الزر الأيمن)
6. انتظر حتى ينتهي الرفع
```

### التوصيل:
```
AD8232 GND  → Arduino GND
AD8232 3.3V → Arduino 3.3V
AD8232 OUT  → Arduino A0
AD8232 LO-  → Arduino Pin 10
AD8232 LO+  → Arduino Pin 11
```

### ماذا سيحدث؟
الأردوينو سيبدأ في:
- ✅ قراءة الإشارة من الحساس
- ✅ استخلاص الميزات
- ✅ إرسال البيانات إلى الكمبيوتر

**ملاحظة:** هذا الكود يعمل بشكل مستمر على الأردوينو!

---

## 🐍 الخطوة 3: برنامج البايثون الرئيسي

### ما هو هذا البرنامج؟
برنامج يقوم بـ:
1. قراءة البيانات من الأردوينو
2. استخدام النموذج المدرب للتنبؤ
3. عرض النتائج
4. حفظ النتائج

### الملف:
📄 **python_arduino_integration.py**

### الخطوات:
```bash
# 1. تثبيت المكتبات
pip install pyserial scikit-learn numpy

# 2. تعديل المنفذ (مهم!)
# افتح الملف وغير هذا السطر:
SERIAL_PORT = 'COM3'  # غيّره إلى منفذك (COM3, COM4, إلخ)

# 3. تشغيل البرنامج
python3 python_arduino_integration.py
```

### ماذا سيحدث؟
```
✅ الاتصال بالأردوينو
✅ قراءة بيانات النبضة الأولى
✅ استخدام النموذج للتنبؤ
✅ عرض النتيجة (طبيعي أو مرض)
✅ عرض درجة الثقة
✅ حفظ النتيجة في ملف
✅ تكرار العملية لكل نبضة جديدة
```

---

## 🔄 ترتيب التشغيل الصحيح:

### المرة الأولى (إعداد):
```
1️⃣ شغّل train_model.py
   ↓
   ينتج: heart_disease_model.pkl
   
2️⃣ رفع كود الأردوينو
   ↓
   الأردوينو يبدأ في قراءة البيانات
   
3️⃣ شغّل python_arduino_integration.py
   ↓
   البرنامج يقرأ من الأردوينو ويتنبأ
```

### المرات اللاحقة (الاستخدام):
```
1️⃣ تأكد من أن الأردوينو مرفوع عليه الكود
   (لا تحتاج لرفعه مرة أخرى)
   
2️⃣ وصّل الأردوينو بـ USB
   
3️⃣ شغّل python_arduino_integration.py
   ↓
   البرنامج يقرأ من الأردوينو ويتنبأ
```

---

## 📂 أين تحط كل ملف؟

### على جهازك (Windows/Mac/Linux):

```
C:\Users\YourName\heart-disease-project\
│
├── AI_Model\
│   ├── train_model.py              ← من GitHub
│   └── heart_disease_model.pkl     ← ينشأ تلقائياً
│
├── Arduino\
│   ├── arduino_ecg_sensor.ino      ← من GitHub
│   └── ARDUINO_SETUP.md            ← من GitHub
│
└── Python_Programs\
    ├── python_arduino_integration.py  ← من GitHub
    └── ecg_results.txt             ← ينشأ تلقائياً
```

### على GitHub:
```
https://github.com/moaz26medical/heart-disease-ai/
│
├── train_model.py
├── arduino_ecg_sensor.ino
├── python_arduino_integration.py
├── ARDUINO_SETUP.md
└── index.html
```

---

## 🎯 ملخص الملفات والوظائف:

| الملف | الوظيفة | متى تشغله | الناتج |
| :--- | :--- | :--- | :--- |
| `train_model.py` | تدريب النموذج | مرة واحدة | `heart_disease_model.pkl` |
| `arduino_ecg_sensor.ino` | قراءة الحساس | رفع مرة واحدة | بيانات مستمرة |
| `python_arduino_integration.py` | التنبؤ والعرض | في كل استخدام | نتائج + ملف |

---

## ⚠️ نقاط مهمة:

### 1. ترتيب التشغيل:
```
❌ لا تشغل python_arduino_integration.py قبل تدريب النموذج
   (سيبحث عن heart_disease_model.pkl ولن يجده)

✅ شغّل train_model.py أولاً
   ثم شغّل python_arduino_integration.py
```

### 2. المنفذ التسلسلي:
```
❌ لا تشغل برنامجين على نفس المنفذ
   (سيحدث تضارب)

✅ تأكد من اختيار المنفذ الصحيح
   في python_arduino_integration.py
```

### 3. النموذج المدرب:
```
❌ لا تحذف heart_disease_model.pkl
   (البرنامج يحتاجه للتنبؤ)

✅ احفظه في نفس المجلد أو عدّل المسار
   في python_arduino_integration.py
```

---

## 🔍 كيفية معرفة المنفذ الصحيح:

### على Windows:
```
1. افتح Device Manager
2. ابحث عن "COM Ports"
3. ستجد "Arduino Uno (COM3)" أو ما شابه
4. استخدم هذا الرقم في البرنامج
```

### على Mac/Linux:
```bash
# اكتب هذا الأمر في Terminal
ls /dev/tty.*

# ستجد شيء مثل:
# /dev/tty.usbserial-1410
```

---

## 📝 خطوات الاستخدام اليومية:

### الصباح (أول مرة):
```bash
# 1. تدريب النموذج (مرة واحدة فقط)
python3 train_model.py

# 2. رفع كود الأردوينو (مرة واحدة فقط)
# افتح Arduino IDE وارفع الكود

# 3. وصّل الأردوينو بـ USB
```

### كل مرة تريد قراءة بيانات:
```bash
# شغّل البرنامج الرئيسي
python3 python_arduino_integration.py

# اضغط Ctrl+C للإيقاف
```

---

## ✅ قائمة التحقق:

قبل البدء:
- [ ] هل ثبّت Python 3؟
- [ ] هل ثبّت Arduino IDE؟
- [ ] هل لديك حساس AD8232؟
- [ ] هل لديك أردوينو Uno؟
- [ ] هل لديك أقطاب كهربائية؟

قبل تشغيل البرنامج:
- [ ] هل شغّلت train_model.py؟
- [ ] هل رفعت كود الأردوينو؟
- [ ] هل وصّلت الأردوينو بـ USB؟
- [ ] هل اخترت المنفذ الصحيح؟
- [ ] هل وضعت الأقطاب على جسمك؟

---

## 🆘 استكشاف الأخطاء:

### المشكلة: "No module named 'serial'"
```bash
# الحل:
pip install pyserial
```

### المشكلة: "heart_disease_model.pkl not found"
```
الحل:
1. تأكد من تشغيل train_model.py أولاً
2. تأكد من أن الملف في نفس المجلد
3. أو عدّل المسار في البرنامج
```

### المشكلة: "Could not open port COM3"
```
الحل:
1. تأكد من توصيل الأردوينو
2. تأكد من المنفذ الصحيح
3. أغلق برنامج Arduino IDE
4. أعد تشغيل البرنامج
```

---

## 📚 ملفات إضافية مفيدة:

- `ARDUINO_SETUP.md` - شرح توصيل الأردوينو
- `train_model.py` - كود تدريب النموذج
- `arduino_ecg_sensor.ino` - كود الأردوينو
- `python_arduino_integration.py` - برنامج البايثون

---

**الآن أنت جاهز للبدء! 🚀**

هل تريد أي توضيح إضافي؟
