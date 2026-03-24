import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = 'medical_system.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 建立 Doctor 表 (加入 is_deleted)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Doctor (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            display_name TEXT NOT NULL,  -- 介面顯示的名稱
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_deleted INTEGER DEFAULT 0
        )
    ''')

    # 建立 Patient 表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Patient (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doctor_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_deleted INTEGER DEFAULT 0,
            FOREIGN KEY (doctor_id) REFERENCES Doctor (id)
        )
    ''')

    # 建立 Photo 表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Photo (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            original_image_path TEXT NOT NULL,
            result_image_path TEXT,
            tumor_type TEXT,
            confidence REAL,
            model_used TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            is_deleted INTEGER DEFAULT 0,
            FOREIGN KEY (patient_id) REFERENCES Patient (id)
        )
    ''')
    conn.commit()
    conn.close()

# --- 醫師 (Doctor) 相關操作 ---
def register_doctor(username, password, display_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    hashed_pw = generate_password_hash(password)
    try:
        cursor.execute("INSERT INTO Doctor (username, password, display_name) VALUES (?, ?, ?)", 
                       (username, hashed_pw, display_name))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # 帳號已存在
    finally:
        conn.close()

def verify_doctor(username, password):
    conn = get_db_connection()
    doctor = conn.execute("SELECT * FROM Doctor WHERE username = ? AND is_deleted = 0", (username,)).fetchone()
    conn.close()
    if doctor and check_password_hash(doctor['password'], password):
        return dict(doctor)
    return None

def update_doctor_display_name(doctor_id, new_name):
    """更新醫師的顯示名稱"""
    conn = get_db_connection()
    conn.execute("UPDATE Doctor SET display_name = ? WHERE id = ?", (new_name, doctor_id))
    conn.commit()
    conn.close()
    
def update_doctor_password(doctor_id, new_password):
    conn = get_db_connection()
    hashed_pw = generate_password_hash(new_password)
    conn.execute("UPDATE Doctor SET password = ? WHERE id = ?", (hashed_pw, doctor_id))
    conn.commit()
    conn.close()

def soft_delete_doctor(doctor_id):
    conn = get_db_connection()
    conn.execute("UPDATE Doctor SET is_deleted = 1 WHERE id = ?", (doctor_id,))
    conn.commit()
    conn.close()

# --- 病患 (Patient) 與 照片 (Photo) 相關操作 ---
def get_or_create_patient(doctor_id, patient_name):
    if not patient_name: return None
    conn = get_db_connection()
    patient = conn.execute("SELECT id FROM Patient WHERE doctor_id = ? AND name = ? AND is_deleted = 0", (doctor_id, patient_name)).fetchone()
    if patient:
        conn.close()
        return patient['id']
    else:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO Patient (doctor_id, name) VALUES (?, ?)", (doctor_id, patient_name))
        conn.commit()
        p_id = cursor.lastrowid
        conn.close()
        return p_id

def add_photo(original_image, result_image, tumor_type, confidence, model_used, patient_id=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO Photo (patient_id, original_image_path, result_image_path, tumor_type, confidence, model_used)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (patient_id, original_image, result_image, tumor_type, confidence, model_used))
    conn.commit()
    conn.close()

def get_doctor_history(doctor_id):
    """取得該醫師所有病患與其照片 (未刪除的)"""
    conn = get_db_connection()
    patients = conn.execute("SELECT * FROM Patient WHERE doctor_id = ? AND is_deleted = 0", (doctor_id,)).fetchall()
    history = []
    for p in patients:
        photos = conn.execute("SELECT * FROM Photo WHERE patient_id = ? AND is_deleted = 0 ORDER BY created_at DESC", (p['id'],)).fetchall()
        history.append({'patient': dict(p), 'photos': [dict(ph) for ph in photos]})
    conn.close()
    return history

def get_photo_by_id(photo_id, doctor_id):
    """取得特定照片，並確保它屬於該醫師的病患"""
    conn = get_db_connection()
    photo = conn.execute('''
        SELECT p.*, pat.name as patient_name 
        FROM Photo p
        JOIN Patient pat ON p.patient_id = pat.id
        WHERE p.id = ? AND pat.doctor_id = ? AND p.is_deleted = 0 AND pat.is_deleted = 0
    ''', (photo_id, doctor_id)).fetchone()
    conn.close()
    return dict(photo) if photo else None

def soft_delete_patient(patient_id, doctor_id):
    conn = get_db_connection()
    # 確保該病患屬於該醫師
    conn.execute("UPDATE Patient SET is_deleted = 1 WHERE id = ? AND doctor_id = ?", (patient_id, doctor_id))
    # 同時軟刪除該病患的所有照片 (Cascade Soft Delete)
    conn.execute("UPDATE Photo SET is_deleted = 1 WHERE patient_id = ?", (patient_id,))
    conn.commit()
    conn.close()

def search_patients_by_name(doctor_id, search_query, limit=5):
    """搜尋該醫師旗下，名稱相近且未被刪除的病患"""
    conn = get_db_connection()
    # 執行 SQL 查詢
    patients = conn.execute(
        "SELECT id, name FROM Patient WHERE doctor_id = ? AND name LIKE ? AND is_deleted = 0 LIMIT ?", 
        (doctor_id, f'%{search_query}%', limit)
    ).fetchall()
    conn.close()
    
    # 直接在這邊轉成字典格式回傳
    return [dict(p) for p in patients]

def update_patient_name(patient_id, doctor_id, new_name):
    """更新病患名稱 (需確保該病患屬於該醫師)"""
    conn = get_db_connection()
    conn.execute("UPDATE Patient SET name = ? WHERE id = ? AND doctor_id = ?", (new_name, patient_id, doctor_id))
    conn.commit()
    conn.close()

def update_photo_patient(photo_id, new_patient_id, doctor_id):
    """將照片轉移給另一個病患 (修改患者)"""
    conn = get_db_connection()
    # 資安防護：先確認這張照片確實屬於該醫師旗下的病患
    photo = conn.execute('''
        SELECT p.id FROM Photo p 
        JOIN Patient pat ON p.patient_id = pat.id 
        WHERE p.id = ? AND pat.doctor_id = ?
    ''', (photo_id, doctor_id)).fetchone()
    
    if photo:
        conn.execute("UPDATE Photo SET patient_id = ? WHERE id = ?", (new_patient_id, photo_id))
        conn.commit()
    conn.close()