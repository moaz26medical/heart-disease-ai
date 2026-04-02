/*
 * =====================================================
 * كود الأردوينو لقراءة حساس AD8232 (ECG Sensor)
 * Arduino Code for AD8232 Heart Rate Sensor
 * =====================================================
 * 
 * الحساس: AD8232 ECG Sensor Module
 * الأردوينو: Arduino Uno / Mega / Nano
 * 
 * الاتصالات:
 * AD8232 GND  --> Arduino GND
 * AD8232 3.3V --> Arduino 3.3V
 * AD8232 OUT  --> Arduino A0 (Analog Pin)
 * AD8232 LO-  --> Arduino Pin 10
 * AD8232 LO+  --> Arduino Pin 11
 * 
 * =====================================================
 */

// ============================================
// 1. تعريف المتغيرات والثوابت
// ============================================

// دبابيس الاتصال
const int ECG_PIN = A0;      // دبوس الإشارة (Analog)
const int LO_MINUS = 10;     // دبوس كشف فقدان الاتصال السالب
const int LO_PLUS = 11;      // دبوس كشف فقدان الاتصال الموجب

// معاملات المعالجة
const int SAMPLE_RATE = 360;      // تردد العينات (Hz) - 360 عينة في الثانية
const int SAMPLES_PER_BEAT = 288; // عدد العينات لكل نبضة (0.8 ثانية)
const int BUFFER_SIZE = 288;      // حجم المخزن المؤقت

// المتغيرات
int ecgBuffer[BUFFER_SIZE];        // مخزن البيانات
int bufferIndex = 0;               // موضع المؤشر الحالي
unsigned long lastSampleTime = 0;  // آخر وقت لأخذ عينة
unsigned long sampleInterval = 1000000 / SAMPLE_RATE;  // الفاصل الزمني بين العينات (ميكروثانية)

// متغيرات الإحصائيات
float maxValue = 0;
float minValue = 1023;
float meanValue = 0;
float stdDeviation = 0;

// ============================================
// 2. دالة الإعداد (Setup)
// ============================================

void setup() {
  // تهيئة الاتصال التسلسلي (Serial Communication)
  Serial.begin(115200);  // معدل البود 115200 (سريع)
  
  // تهيئة الدبابيس
  pinMode(LO_MINUS, INPUT);  // كشف فقدان الاتصال
  pinMode(LO_PLUS, INPUT);   // كشف فقدان الاتصال
  pinMode(ECG_PIN, INPUT);   // قراءة الإشارة
  
  // رسالة البداية
  delay(1000);
  Serial.println("=====================================");
  Serial.println("🏥 نظام قراءة حساس AD8232 ECG");
  Serial.println("AD8232 ECG Sensor Reading System");
  Serial.println("=====================================");
  Serial.print("تردد العينات: ");
  Serial.print(SAMPLE_RATE);
  Serial.println(" Hz");
  Serial.print("عدد العينات لكل نبضة: ");
  Serial.println(SAMPLES_PER_BEAT);
  Serial.println("جاري البدء...");
  Serial.println("=====================================\n");
  
  // تهيئة المخزن المؤقت
  for (int i = 0; i < BUFFER_SIZE; i++) {
    ecgBuffer[i] = 0;
  }
  
  lastSampleTime = micros();
}

// ============================================
// 3. دالة الحلقة الرئيسية (Loop)
// ============================================

void loop() {
  // التحقق من فقدان الاتصال
  if (digitalRead(LO_MINUS) == 1 || digitalRead(LO_PLUS) == 1) {
    Serial.println("⚠️ تحذير: فقدان الاتصال! تحقق من توصيل الحساس");
    delay(1000);
    return;
  }
  
  // قراءة العينة في الوقت المناسب
  unsigned long currentTime = micros();
  if (currentTime - lastSampleTime >= sampleInterval) {
    // قراءة قيمة الحساس
    int ecgValue = analogRead(ECG_PIN);
    
    // إضافة القيمة إلى المخزن المؤقت
    ecgBuffer[bufferIndex] = ecgValue;
    bufferIndex++;
    
    // طباعة القيمة الحالية (للمراقبة)
    Serial.print(ecgValue);
    Serial.print(" ");
    
    // عندما يمتلئ المخزن المؤقت (نبضة واحدة كاملة)
    if (bufferIndex >= BUFFER_SIZE) {
      Serial.println();  // سطر جديد
      
      // معالجة النبضة الكاملة
      processHeartbeat();
      
      // إعادة تعيين المؤشر
      bufferIndex = 0;
    }
    
    lastSampleTime = currentTime;
  }
}

// ============================================
// 4. معالجة النبضة الكاملة
// ============================================

void processHeartbeat() {
  Serial.println("\n--- نبضة قلب جديدة ---");
  
  // حساب الإحصائيات
  calculateStatistics();
  
  // طباعة النتائج
  printStatistics();
  
  // استخلاص الميزات
  float features[10];
  extractFeatures(features);
  
  // طباعة الميزات
  printFeatures(features);
  
  // إرسال البيانات إلى البايثون (للتنبؤ)
  sendDataToPython(features);
  
  Serial.println("-------------------\n");
}

// ============================================
// 5. حساب الإحصائيات
// ============================================

void calculateStatistics() {
  maxValue = -1023;
  minValue = 1023;
  float sum = 0;
  float sumSquares = 0;
  
  // حساب المتوسط والقيم العظمى والصغرى
  for (int i = 0; i < BUFFER_SIZE; i++) {
    int value = ecgBuffer[i];
    
    if (value > maxValue) maxValue = value;
    if (value < minValue) minValue = value;
    
    sum += value;
    sumSquares += value * value;
  }
  
  // المتوسط
  meanValue = sum / BUFFER_SIZE;
  
  // الانحراف المعياري
  float variance = (sumSquares / BUFFER_SIZE) - (meanValue * meanValue);
  stdDeviation = sqrt(variance);
}

// ============================================
// 6. طباعة الإحصائيات
// ============================================

void printStatistics() {
  Serial.print("📊 الإحصائيات:\n");
  Serial.print("   القيمة العظمى: ");
  Serial.println(maxValue);
  
  Serial.print("   القيمة الصغرى: ");
  Serial.println(minValue);
  
  Serial.print("   المتوسط: ");
  Serial.println(meanValue);
  
  Serial.print("   الانحراف المعياري: ");
  Serial.println(stdDeviation);
  
  Serial.print("   النطاق: ");
  Serial.println(maxValue - minValue);
}

// ============================================
// 7. استخلاص الميزات (Feature Extraction)
// ============================================

void extractFeatures(float features[]) {
  // الميزة 1: القيمة العظمى
  features[0] = maxValue;
  
  // الميزة 2: القيمة الصغرى
  features[1] = minValue;
  
  // الميزة 3: المتوسط
  features[2] = meanValue;
  
  // الميزة 4: الانحراف المعياري
  features[3] = stdDeviation;
  
  // الميزة 5: الطاقة
  float energy = 0;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    energy += ecgBuffer[i] * ecgBuffer[i];
  }
  features[4] = energy;
  
  // الميزة 6: الانحراف (Skewness)
  float skewness = 0;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    float diff = ecgBuffer[i] - meanValue;
    skewness += diff * diff * diff;
  }
  skewness = skewness / (BUFFER_SIZE * stdDeviation * stdDeviation * stdDeviation);
  features[5] = skewness;
  
  // الميزة 7: التفرطح (Kurtosis)
  float kurtosis = 0;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    float diff = ecgBuffer[i] - meanValue;
    kurtosis += diff * diff * diff * diff;
  }
  kurtosis = kurtosis / (BUFFER_SIZE * stdDeviation * stdDeviation * stdDeviation * stdDeviation);
  features[6] = kurtosis;
  
  // الميزة 8: معدل التغيير
  float rateOfChange = 0;
  for (int i = 1; i < BUFFER_SIZE; i++) {
    int diff = ecgBuffer[i] - ecgBuffer[i-1];
    rateOfChange += (diff < 0) ? -diff : diff;  // القيمة المطلقة
  }
  rateOfChange = rateOfChange / BUFFER_SIZE;
  features[7] = rateOfChange;
  
  // الميزة 9: متوسط منطقة ST (الأهم للكشف عن المرض)
  int midPoint = BUFFER_SIZE / 2;
  int stStart = (int)(midPoint * 0.85);
  int stEnd = (int)(midPoint * 1.15);
  float stMean = 0;
  for (int i = stStart; i < stEnd; i++) {
    stMean += ecgBuffer[i];
  }
  stMean = stMean / (stEnd - stStart);
  features[8] = stMean;
  
  // الميزة 10: عدد تقاطعات الصفر
  float zeroCrossings = 0;
  for (int i = 1; i < BUFFER_SIZE; i++) {
    if ((ecgBuffer[i-1] - meanValue) * (ecgBuffer[i] - meanValue) < 0) {
      zeroCrossings++;
    }
  }
  features[9] = zeroCrossings;
}

// ============================================
// 8. طباعة الميزات
// ============================================

void printFeatures(float features[]) {
  Serial.println("🔍 الميزات المستخلصة:");
  Serial.print("   1. القيمة العظمى: ");
  Serial.println(features[0]);
  Serial.print("   2. القيمة الصغرى: ");
  Serial.println(features[1]);
  Serial.print("   3. المتوسط: ");
  Serial.println(features[2]);
  Serial.print("   4. الانحراف المعياري: ");
  Serial.println(features[3]);
  Serial.print("   5. الطاقة: ");
  Serial.println(features[4]);
  Serial.print("   6. الانحراف (Skewness): ");
  Serial.println(features[5]);
  Serial.print("   7. التفرطح (Kurtosis): ");
  Serial.println(features[6]);
  Serial.print("   8. معدل التغيير: ");
  Serial.println(features[7]);
  Serial.print("   9. متوسط ST (الأهم): ");
  Serial.println(features[8]);
  Serial.print("   10. تقاطعات الصفر: ");
  Serial.println(features[9]);
}

// ============================================
// 9. إرسال البيانات إلى البايثون
// ============================================

void sendDataToPython(float features[]) {
  // صيغة الإرسال: HEARTBEAT:f1,f2,f3,...,f10
  Serial.print("📤 إرسال البيانات: HEARTBEAT:");
  for (int i = 0; i < 10; i++) {
    Serial.print(features[i]);
    if (i < 9) Serial.print(",");
  }
  Serial.println();
}

// ============================================
// 10. ملاحظة: الدوال المساعدة
// ============================================

// ملاحظة: نستخدم الدوال المدمجة في Arduino:
// - abs() - القيمة المطلقة (مدمجة)
// - sqrt() - الجذر التربيعي (مدمجة)
// لا نحتاج لكتابة دوال جديدة!

/*
 * =====================================================
 * ملاحظات مهمة:
 * =====================================================
 * 
 * 1. معدل البود (Baud Rate):
 *    - استخدمنا 115200 للسرعة العالية
 *    - تأكد من اختيار نفس المعدل في Serial Monitor
 * 
 * 2. توصيل الحساس:
 *    - AD8232 GND  --> Arduino GND
 *    - AD8232 3.3V --> Arduino 3.3V (أو 5V مع مقاوم)
 *    - AD8232 OUT  --> Arduino A0
 *    - AD8232 LO-  --> Arduino Pin 10
 *    - AD8232 LO+  --> Arduino Pin 11
 * 
 * 3. الأقطاب الكهربائية:
 *    - الأحمر (RA): الذراع اليمنى
 *    - الأسود (LA): الذراع اليسرى
 *    - الأبيض (RL): الرجل اليمنى (مرجع)
 * 
 * 4. تردد العينات:
 *    - 360 Hz كافٍ لقراءة ECG
 *    - يمكن تقليله إلى 250 Hz لتوفير الموارد
 * 
 * 5. المعايرة:
 *    - قد تحتاج لمعايرة الحساس قبل الاستخدام
 *    - اقرأ قيم الخط الأساسي بدون توصيل أقطاب
 * 
 * 6. التنبؤ بالمرض:
 *    - الميزة 9 (متوسط ST) هي الأهم
 *    - إذا كانت منخفضة جداً = احتمال نقص تروية
 * 
 * =====================================================
 */
