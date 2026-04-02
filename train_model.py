"""
نموذج الذكاء الاصطناعي لتشخيص أمراض القلب
Heart Disease AI Diagnosis Model
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. توليد بيانات تدريبية محاكاة
# ============================================

def generate_normal_heartbeats(num_beats=300, fs=360):
    """
    توليد نبضات قلب طبيعية (Normal Sinus Rhythm)
    fs: تردد العينات (Sampling Frequency)
    """
    heartbeats = []
    beat_duration = int(fs * 0.8)  # 0.8 ثانية لكل نبضة
    
    for _ in range(num_beats):
        t = np.linspace(0, 0.8, beat_duration)
        
        # موجة P (انقباض الأذينين)
        p_wave = 0.15 * np.exp(-((t - 0.1) ** 2) / 0.01)
        
        # مركب QRS (انقباض البطينين)
        q_wave = -0.2 * np.exp(-((t - 0.3) ** 2) / 0.002)
        r_wave = 1.0 * np.exp(-((t - 0.32) ** 2) / 0.002)
        s_wave = -0.3 * np.exp(-((t - 0.34) ** 2) / 0.002)
        
        # موجة T (انبساط البطينين)
        t_wave = 0.3 * np.exp(-((t - 0.5) ** 2) / 0.015)
        
        # مقطع ST (طبيعي = على خط الصفر)
        st_segment = np.zeros_like(t)
        
        # الإشارة الكاملة
        ecg = p_wave + q_wave + r_wave + s_wave + t_wave + st_segment
        
        # إضافة ضجيج طفيف
        noise = np.random.normal(0, 0.05, len(ecg))
        ecg = ecg + noise
        
        heartbeats.append(ecg)
    
    return np.array(heartbeats)


def generate_ischemia_heartbeats(num_beats=300, fs=360):
    """
    توليد نبضات قلب مع نقص تروية (Ischemia/Atherosclerosis)
    العلامات: انخفاض ST، انقلاب موجة T
    """
    heartbeats = []
    beat_duration = int(fs * 0.8)
    
    for _ in range(num_beats):
        t = np.linspace(0, 0.8, beat_duration)
        
        # موجة P (عادية)
        p_wave = 0.15 * np.exp(-((t - 0.1) ** 2) / 0.01)
        
        # مركب QRS (قد يكون أقل قليلاً)
        q_wave = -0.15 * np.exp(-((t - 0.3) ** 2) / 0.002)
        r_wave = 0.8 * np.exp(-((t - 0.32) ** 2) / 0.002)
        s_wave = -0.25 * np.exp(-((t - 0.34) ** 2) / 0.002)
        
        # موجة T (مقلوبة في حالة المرض)
        t_wave = -0.3 * np.exp(-((t - 0.5) ** 2) / 0.015)  # سالبة بدلاً من موجبة
        
        # مقطع ST (منخفض في حالة نقص التروية)
        st_segment = np.where((t > 0.34) & (t < 0.5), -0.25, 0)
        
        # الإشارة الكاملة
        ecg = p_wave + q_wave + r_wave + s_wave + t_wave + st_segment
        
        # إضافة ضجيج
        noise = np.random.normal(0, 0.05, len(ecg))
        ecg = ecg + noise
        
        heartbeats.append(ecg)
    
    return np.array(heartbeats)


# ============================================
# 2. استخلاص الميزات (Feature Extraction)
# ============================================

def extract_features(heartbeat):
    """
    استخلاص ميزات من نبضة القلب الواحدة
    الميزات تساعد الذكاء الاصطناعي على التمييز بين الحالات
    """
    features = []
    
    # 1. القيمة العظمى (Maximum value)
    features.append(np.max(heartbeat))
    
    # 2. القيمة الصغرى (Minimum value)
    features.append(np.min(heartbeat))
    
    # 3. المتوسط (Mean)
    features.append(np.mean(heartbeat))
    
    # 4. الانحراف المعياري (Standard deviation)
    features.append(np.std(heartbeat))
    
    # 5. الطاقة (Energy)
    features.append(np.sum(heartbeat ** 2))
    
    # 6. الانحراف (Skewness)
    features.append(np.mean((heartbeat - np.mean(heartbeat)) ** 3) / (np.std(heartbeat) ** 3))
    
    # 7. التفرطح (Kurtosis)
    features.append(np.mean((heartbeat - np.mean(heartbeat)) ** 4) / (np.std(heartbeat) ** 4))
    
    # 8. معدل التغيير (Rate of change)
    diff = np.diff(heartbeat)
    features.append(np.mean(np.abs(diff)))
    
    # 9. الانحدار الأفقي (Horizontal slope) - مهم جداً لاكتشاف ST
    mid_point = len(heartbeat) // 2
    st_region = heartbeat[int(mid_point * 0.85):int(mid_point * 1.15)]
    features.append(np.mean(st_region))  # متوسط منطقة ST
    
    # 10. عدد التقاطعات مع الصفر (Zero crossings)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(heartbeat)))) / 2
    features.append(zero_crossings)
    
    return np.array(features)


# ============================================
# 3. إعداد البيانات للتدريب
# ============================================

print("=" * 60)
print("🏥 نموذج الذكاء الاصطناعي لتشخيص أمراض القلب")
print("=" * 60)

print("\n📊 جاري توليد البيانات التدريبية...")

# توليد البيانات
normal_beats = generate_normal_heartbeats(num_beats=300)
ischemia_beats = generate_ischemia_heartbeats(num_beats=300)

print(f"✅ تم توليد {len(normal_beats)} نبضة قلب طبيعية")
print(f"✅ تم توليد {len(ischemia_beats)} نبضة قلب مع نقص تروية")

# استخلاص الميزات
print("\n🔍 جاري استخلاص الميزات من الإشارات...")
normal_features = np.array([extract_features(beat) for beat in normal_beats])
ischemia_features = np.array([extract_features(beat) for beat in ischemia_beats])

print(f"✅ تم استخلاص {len(normal_features[0])} ميزة من كل نبضة")

# دمج البيانات
X = np.vstack([normal_features, ischemia_features])
y = np.hstack([np.zeros(len(normal_features)), np.ones(len(ischemia_features))])

print(f"✅ إجمالي البيانات: {len(X)} نبضة")
print(f"   - الفئة 0 (طبيعي): {np.sum(y == 0)} نبضة")
print(f"   - الفئة 1 (نقص تروية): {np.sum(y == 1)} نبضة")

# تقسيم البيانات إلى تدريب واختبار
print("\n📈 تقسيم البيانات...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ بيانات التدريب: {len(X_train)} نبضة")
print(f"✅ بيانات الاختبار: {len(X_test)} نبضة")

# ============================================
# 4. تدريب النموذج
# ============================================

print("\n🤖 جاري تدريب نموذج Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,  # عدد الأشجار
    max_depth=15,      # أقصى عمق للشجرة
    random_state=42,
    n_jobs=-1          # استخدام كل المعالجات المتاحة
)

model.fit(X_train, y_train)
print("✅ تم تدريب النموذج بنجاح!")

# ============================================
# 5. تقييم النموذج
# ============================================

print("\n📊 تقييم النموذج على بيانات الاختبار...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"📈 نتائج الأداء:")
print(f"{'='*60}")
print(f"✅ الدقة (Accuracy):        {accuracy:.2%}")
print(f"✅ الدقة الموجبة (Precision): {precision:.2%}")
print(f"✅ الاستدعاء (Recall):      {recall:.2%}")
print(f"✅ درجة F1 (F1-Score):      {f1:.2%}")
print(f"{'='*60}\n")

# ============================================
# 6. رسم النتائج
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('🏥 نتائج تدريب نموذج تشخيص أمراض القلب', fontsize=16, fontweight='bold')

# 1. مصفوفة الالتباس (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], cbar=False)
axes[0, 0].set_title('مصفوفة الالتباس (Confusion Matrix)')
axes[0, 0].set_ylabel('القيمة الفعلية')
axes[0, 0].set_xlabel('القيمة المتنبأ بها')
axes[0, 0].set_xticklabels(['طبيعي', 'نقص تروية'])
axes[0, 0].set_yticklabels(['طبيعي', 'نقص تروية'])

# 2. أهمية الميزات (Feature Importance)
feature_importance = model.feature_importances_
feature_names = [f'الميزة {i+1}' for i in range(len(feature_importance))]
axes[0, 1].barh(feature_names, feature_importance, color='steelblue')
axes[0, 1].set_title('أهمية الميزات (Feature Importance)')
axes[0, 1].set_xlabel('درجة الأهمية')

# 3. مقارنة المقاييس
metrics = ['الدقة', 'الدقة الموجبة', 'الاستدعاء', 'F1-Score']
scores = [accuracy, precision, recall, f1]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
axes[1, 0].bar(metrics, scores, color=colors, alpha=0.7)
axes[1, 0].set_title('مقارنة المقاييس')
axes[1, 0].set_ylabel('الدرجة')
axes[1, 0].set_ylim([0, 1])
for i, v in enumerate(scores):
    axes[1, 0].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')

# 4. توزيع التنبؤات
axes[1, 1].hist(model.predict_proba(X_test)[:, 1], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
axes[1, 1].set_title('توزيع احتمالية التنبؤ')
axes[1, 1].set_xlabel('احتمالية نقص التروية')
axes[1, 1].set_ylabel('التكرار')

plt.tight_layout()
plt.savefig('/home/ubuntu/model_results.png', dpi=300, bbox_inches='tight')
print("✅ تم حفظ الرسوم البيانية في: /home/ubuntu/model_results.png")
plt.show()

# ============================================
# 7. اختبار على إشارات جديدة
# ============================================

print("\n" + "="*60)
print("🔬 اختبار النموذج على إشارات جديدة")
print("="*60)

# توليد إشارات جديدة للاختبار
new_normal = generate_normal_heartbeats(num_beats=5)
new_ischemia = generate_ischemia_heartbeats(num_beats=5)

# استخلاص الميزات
new_normal_features = np.array([extract_features(beat) for beat in new_normal])
new_ischemia_features = np.array([extract_features(beat) for beat in new_ischemia])

# التنبؤ
normal_predictions = model.predict(new_normal_features)
normal_probabilities = model.predict_proba(new_normal_features)

ischemia_predictions = model.predict(new_ischemia_features)
ischemia_probabilities = model.predict_proba(new_ischemia_features)

print("\n📋 نتائج الاختبار على إشارات طبيعية:")
for i, (pred, prob) in enumerate(zip(normal_predictions, normal_probabilities)):
    status = "✅ طبيعي" if pred == 0 else "⚠️ نقص تروية"
    confidence = prob[int(pred)]
    print(f"   النبضة {i+1}: {status} (ثقة: {confidence:.2%})")

print("\n📋 نتائج الاختبار على إشارات نقص تروية:")
for i, (pred, prob) in enumerate(zip(ischemia_predictions, ischemia_probabilities)):
    status = "✅ طبيعي" if pred == 0 else "⚠️ نقص تروية"
    confidence = prob[int(pred)]
    print(f"   النبضة {i+1}: {status} (ثقة: {confidence:.2%})")

# ============================================
# 8. حفظ النموذج
# ============================================

import pickle

model_path = '/home/ubuntu/heart_disease_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"\n✅ تم حفظ النموذج في: {model_path}")

print("\n" + "="*60)
print("🎉 انتهى التدريب بنجاح!")
print("="*60)
print("\n📌 الخطوات التالية:")
print("   1. ربط حساس AD8232 مع الأردوينو")
print("   2. إرسال الإشارات الحقيقية إلى البايثون")
print("   3. استخدام النموذج المدرب للتنبؤ")
print("   4. عرض النتائج على الواجهة الإلكترونية")
