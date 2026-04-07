"""
=====================================================
نظام قاعدة البيانات - حفظ نتائج التشخيص
Database System - Save Diagnosis Results
=====================================================

برنامج لإدارة قاعدة بيانات SQLite لحفظ نتائج التشخيص
"""

import sqlite3
import json
from datetime import datetime
import os

# ============================================
# فئة إدارة قاعدة البيانات
# ============================================

class HeartDiseaseDatabase:
    def __init__(self, db_name='heart_disease_results.db'):
        """تهيئة قاعدة البيانات"""
        self.db_name = db_name
        self.connection = None
        self.cursor = None
        
        # إنشاء قاعدة البيانات إذا لم تكن موجودة
        self.create_database()
    
    def create_database(self):
        """إنشاء قاعدة البيانات والجداول"""
        try:
            self.connection = sqlite3.connect(self.db_name)
            self.cursor = self.connection.cursor()
            
            # جدول المرضى
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER,
                    gender TEXT,
                    email TEXT,
                    phone TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # جدول الاختبارات
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS tests (
                    test_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    test_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    heartbeat_count INTEGER,
                    avg_heart_rate REAL,
                    prediction TEXT,
                    confidence REAL,
                    features TEXT,
                    notes TEXT,
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                )
            ''')
            
            # جدول النتائج التفصيلية
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_id INTEGER NOT NULL,
                    heartbeat_number INTEGER,
                    max_value REAL,
                    min_value REAL,
                    mean_value REAL,
                    std_deviation REAL,
                    energy REAL,
                    skewness REAL,
                    kurtosis REAL,
                    rate_of_change REAL,
                    st_mean REAL,
                    zero_crossings REAL,
                    prediction TEXT,
                    confidence REAL,
                    FOREIGN KEY (test_id) REFERENCES tests(test_id)
                )
            ''')
            
            self.connection.commit()
            print("✅ تم إنشاء قاعدة البيانات بنجاح!")
            
        except sqlite3.Error as e:
            print(f"❌ خطأ في إنشاء قاعدة البيانات: {e}")
    
    # ============================================
    # عمليات المرضى
    # ============================================
    
    def add_patient(self, name, age=None, gender=None, email=None, phone=None):
        """إضافة مريض جديد"""
        try:
            self.cursor.execute('''
                INSERT INTO patients (name, age, gender, email, phone)
                VALUES (?, ?, ?, ?, ?)
            ''', (name, age, gender, email, phone))
            
            self.connection.commit()
            patient_id = self.cursor.lastrowid
            print(f"✅ تم إضافة المريض: {name} (ID: {patient_id})")
            return patient_id
            
        except sqlite3.Error as e:
            print(f"❌ خطأ في إضافة المريض: {e}")
            return None
    
    def get_patient(self, patient_id):
        """الحصول على بيانات المريض"""
        try:
            self.cursor.execute('SELECT * FROM patients WHERE patient_id = ?', (patient_id,))
            patient = self.cursor.fetchone()
            return patient
        except sqlite3.Error as e:
            print(f"❌ خطأ: {e}")
            return None
    
    def get_all_patients(self):
        """الحصول على جميع المرضى"""
        try:
            self.cursor.execute('SELECT * FROM patients')
            patients = self.cursor.fetchall()
            return patients
        except sqlite3.Error as e:
            print(f"❌ خطأ: {e}")
            return None
    
    def update_patient(self, patient_id, name=None, age=None, gender=None, email=None, phone=None):
        """تحديث بيانات المريض"""
        try:
            updates = []
            values = []
            
            if name is not None:
                updates.append("name = ?")
                values.append(name)
            if age is not None:
                updates.append("age = ?")
                values.append(age)
            if gender is not None:
                updates.append("gender = ?")
                values.append(gender)
            if email is not None:
                updates.append("email = ?")
                values.append(email)
            if phone is not None:
                updates.append("phone = ?")
                values.append(phone)
            
            if not updates:
                return False
            
            values.append(patient_id)
            query = f"UPDATE patients SET {', '.join(updates)} WHERE patient_id = ?"
            
            self.cursor.execute(query, values)
            self.connection.commit()
            print(f"✅ تم تحديث بيانات المريض: {patient_id}")
            return True
            
        except sqlite3.Error as e:
            print(f"❌ خطأ: {e}")
            return False
    
    def delete_patient(self, patient_id):
        """حذف مريض"""
        try:
            self.cursor.execute('DELETE FROM patients WHERE patient_id = ?', (patient_id,))
            self.connection.commit()
            print(f"✅ تم حذف المريض: {patient_id}")
            return True
        except sqlite3.Error as e:
            print(f"❌ خطأ: {e}")
            return False
    
    # ============================================
    # عمليات الاختبارات
    # ============================================
    
    def add_test(self, patient_id, heartbeat_count, avg_heart_rate, prediction, confidence, features, notes=None):
        """إضافة اختبار جديد"""
        try:
            features_json = json.dumps(features)
            
            self.cursor.execute('''
                INSERT INTO tests (patient_id, heartbeat_count, avg_heart_rate, prediction, confidence, features, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (patient_id, heartbeat_count, avg_heart_rate, prediction, confidence, features_json, notes))
            
            self.connection.commit()
            test_id = self.cursor.lastrowid
            print(f"✅ تم إضافة الاختبار: {test_id}")
            return test_id
            
        except sqlite3.Error as e:
            print(f"❌ خطأ: {e}")
            return None
    
    def get_test(self, test_id):
        """الحصول على بيانات الاختبار"""
        try:
            self.cursor.execute('SELECT * FROM tests WHERE test_id = ?', (test_id,))
            test = self.cursor.fetchone()
            return test
        except sqlite3.Error as e:
            print(f"❌ خطأ: {e}")
            return None
    
    def get_patient_tests(self, patient_id):
        """الحصول على جميع اختبارات المريض"""
        try:
            self.cursor.execute('SELECT * FROM tests WHERE patient_id = ? ORDER BY test_date DESC', (patient_id,))
            tests = self.cursor.fetchall()
            return tests
        except sqlite3.Error as e:
            print(f"❌ خطأ: {e}")
            return None
    
    def get_all_tests(self):
        """الحصول على جميع الاختبارات"""
        try:
            self.cursor.execute('SELECT * FROM tests ORDER BY test_date DESC')
            tests = self.cursor.fetchall()
            return tests
        except sqlite3.Error as e:
            print(f"❌ خطأ: {e}")
            return None
    
    # ============================================
    # عمليات النتائج التفصيلية
    # ============================================
    
    def add_result(self, test_id, heartbeat_number, features, prediction, confidence):
        """إضافة نتيجة تفصيلية"""
        try:
            self.cursor.execute('''
                INSERT INTO results 
                (test_id, heartbeat_number, max_value, min_value, mean_value, std_deviation, 
                 energy, skewness, kurtosis, rate_of_change, st_mean, zero_crossings, 
                 prediction, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                test_id, heartbeat_number,
                features[0], features[1], features[2], features[3],
                features[4], features[5], features[6], features[7],
                features[8], features[9],
                prediction, confidence
            ))
            
            self.connection.commit()
            result_id = self.cursor.lastrowid
            return result_id
            
        except sqlite3.Error as e:
            print(f"❌ خطأ: {e}")
            return None
    
    def get_test_results(self, test_id):
        """الحصول على جميع نتائج الاختبار"""
        try:
            self.cursor.execute('SELECT * FROM results WHERE test_id = ? ORDER BY heartbeat_number', (test_id,))
            results = self.cursor.fetchall()
            return results
        except sqlite3.Error as e:
            print(f"❌ خطأ: {e}")
            return None
    
    # ============================================
    # الإحصائيات والتقارير
    # ============================================
    
    def get_patient_statistics(self, patient_id):
        """الحصول على إحصائيات المريض"""
        try:
            # عدد الاختبارات
            self.cursor.execute('SELECT COUNT(*) FROM tests WHERE patient_id = ?', (patient_id,))
            test_count = self.cursor.fetchone()[0]
            
            # عدد النتائج الإيجابية
            self.cursor.execute(
                'SELECT COUNT(*) FROM tests WHERE patient_id = ? AND prediction = "مرض"',
                (patient_id,)
            )
            positive_count = self.cursor.fetchone()[0]
            
            # متوسط الثقة
            self.cursor.execute(
                'SELECT AVG(confidence) FROM tests WHERE patient_id = ?',
                (patient_id,)
            )
            avg_confidence = self.cursor.fetchone()[0]
            
            return {
                'test_count': test_count,
                'positive_count': positive_count,
                'negative_count': test_count - positive_count,
                'avg_confidence': avg_confidence
            }
            
        except sqlite3.Error as e:
            print(f"❌ خطأ: {e}")
            return None
    
    def get_system_statistics(self):
        """الحصول على إحصائيات النظام"""
        try:
            # عدد المرضى
            self.cursor.execute('SELECT COUNT(*) FROM patients')
            patient_count = self.cursor.fetchone()[0]
            
            # عدد الاختبارات
            self.cursor.execute('SELECT COUNT(*) FROM tests')
            test_count = self.cursor.fetchone()[0]
            
            # عدد النتائج الإيجابية
            self.cursor.execute('SELECT COUNT(*) FROM tests WHERE prediction = "مرض"')
            positive_count = self.cursor.fetchone()[0]
            
            # متوسط الثقة
            self.cursor.execute('SELECT AVG(confidence) FROM tests')
            avg_confidence = self.cursor.fetchone()[0]
            
            return {
                'patient_count': patient_count,
                'test_count': test_count,
                'positive_count': positive_count,
                'negative_count': test_count - positive_count,
                'avg_confidence': avg_confidence
            }
            
        except sqlite3.Error as e:
            print(f"❌ خطأ: {e}")
            return None
    
    def generate_report(self, patient_id, output_file=None):
        """إنشاء تقرير شامل للمريض"""
        try:
            patient = self.get_patient(patient_id)
            if not patient:
                print(f"❌ المريض غير موجود: {patient_id}")
                return None
            
            tests = self.get_patient_tests(patient_id)
            stats = self.get_patient_statistics(patient_id)
            
            report = f"""
{'='*70}
📋 تقرير شامل للمريض
{'='*70}

👤 بيانات المريض:
───────────────────────────────────────────────────────────────────
الاسم: {patient[1]}
العمر: {patient[2]} سنة
الجنس: {patient[3]}
البريد الإلكتروني: {patient[4]}
الهاتف: {patient[5]}
تاريخ التسجيل: {patient[6]}

📊 الإحصائيات:
───────────────────────────────────────────────────────────────────
عدد الاختبارات: {stats['test_count']}
النتائج الإيجابية (مرض): {stats['positive_count']}
النتائج السلبية (طبيعي): {stats['negative_count']}
متوسط الثقة: {stats['avg_confidence']:.2%}

📈 سجل الاختبارات:
───────────────────────────────────────────────────────────────────
"""
            
            if tests:
                for i, test in enumerate(tests, 1):
                    report += f"""
اختبار #{i}:
  التاريخ: {test[2]}
  عدد النبضات: {test[3]}
  متوسط ضربات القلب: {test[4]} نبضة/دقيقة
  النتيجة: {test[5]}
  درجة الثقة: {test[6]:.2%}
  ملاحظات: {test[8] or 'لا توجد'}
"""
            
            report += f"\n{'='*70}\n"
            
            # حفظ التقرير
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"✅ تم حفظ التقرير في: {output_file}")
            
            return report
            
        except Exception as e:
            print(f"❌ خطأ: {e}")
            return None
    
    def close(self):
        """إغلاق الاتصال بقاعدة البيانات"""
        if self.connection:
            self.connection.close()
            print("✅ تم إغلاق قاعدة البيانات")


# ============================================
# برنامج اختبار
# ============================================

def test_database():
    """اختبار قاعدة البيانات"""
    print("\n" + "="*70)
    print("🧪 اختبار نظام قاعدة البيانات")
    print("="*70 + "\n")
    
    # إنشاء قاعدة البيانات
    db = HeartDiseaseDatabase('test_heart_disease.db')
    
    # إضافة مريض
    print("\n1️⃣ إضافة مريض:")
    patient_id = db.add_patient(
        name="أحمد محمد",
        age=45,
        gender="ذكر",
        email="ahmed@example.com",
        phone="0123456789"
    )
    
    # إضافة اختبار
    print("\n2️⃣ إضافة اختبار:")
    features = [1.0, -0.5, 0.2, 0.3, 5.0, 0.1, 0.2, 0.15, -0.25, 3.0]
    test_id = db.add_test(
        patient_id=patient_id,
        heartbeat_count=5,
        avg_heart_rate=72,
        prediction="طبيعي",
        confidence=0.95,
        features=features,
        notes="اختبار عادي"
    )
    
    # إضافة نتائج تفصيلية
    print("\n3️⃣ إضافة نتائج تفصيلية:")
    for i in range(5):
        db.add_result(
            test_id=test_id,
            heartbeat_number=i+1,
            features=features,
            prediction="طبيعي",
            confidence=0.95
        )
    
    # الحصول على إحصائيات المريض
    print("\n4️⃣ إحصائيات المريض:")
    stats = db.get_patient_statistics(patient_id)
    print(f"   عدد الاختبارات: {stats['test_count']}")
    print(f"   النتائج الإيجابية: {stats['positive_count']}")
    print(f"   النتائج السلبية: {stats['negative_count']}")
    print(f"   متوسط الثقة: {stats['avg_confidence']:.2%}")
    
    # الحصول على إحصائيات النظام
    print("\n5️⃣ إحصائيات النظام:")
    system_stats = db.get_system_statistics()
    print(f"   عدد المرضى: {system_stats['patient_count']}")
    print(f"   عدد الاختبارات: {system_stats['test_count']}")
    print(f"   النتائج الإيجابية: {system_stats['positive_count']}")
    
    # إنشاء تقرير
    print("\n6️⃣ إنشاء تقرير:")
    report = db.generate_report(patient_id, 'patient_report.txt')
    print(report)
    
    # إغلاق قاعدة البيانات
    db.close()
    
    print("\n" + "="*70)
    print("✅ انتهى الاختبار بنجاح!")
    print("="*70 + "\n")


if __name__ == '__main__':
    test_database()

"""
=====================================================
كيفية الاستخدام:
=====================================================

1. إنشاء قاعدة بيانات:
   from database_system import HeartDiseaseDatabase
   db = HeartDiseaseDatabase('heart_disease.db')

2. إضافة مريض:
   patient_id = db.add_patient('أحمد', age=45, gender='ذكر')

3. إضافة اختبار:
   test_id = db.add_test(patient_id, 5, 72, 'طبيعي', 0.95, features)

4. الحصول على إحصائيات:
   stats = db.get_patient_statistics(patient_id)

5. إنشاء تقرير:
   report = db.generate_report(patient_id, 'report.txt')

6. إغلاق قاعدة البيانات:
   db.close()

=====================================================
"""
