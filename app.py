import os
import sys
import torch
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import shutil
from classification.prediction import load_model, preprocess_image, predict_single_model, CLASS_NAMES
import jwt
import datetime
from functools import wraps
from flask import make_response, session, jsonify # 處理 Cookie 等
from db import init_db, register_doctor, update_doctor_display_name, verify_doctor, get_or_create_patient, add_photo, get_doctor_history, get_photo_by_id, update_doctor_password, soft_delete_patient, soft_delete_doctor, search_patients_by_name, update_patient_name, update_photo_patient

# --- Path Setup ---
# Resolve the absolute path of this file's directory and register sub-modules
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(BASE_DIR, 'classification'))
sys.path.append(os.path.join(BASE_DIR, 'swin_unet'))
from swin_unet.interface import predict_and_save

app = Flask(__name__)
init_db()
app.config['SECRET_KEY'] = 'medical_super_secret_key'

# --- Folder Configuration ---
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
MODELS_DIR = os.path.join(BASE_DIR, 'models/YOLOv8')
UNET_MODELS_DIR = os.path.join(BASE_DIR, 'models/SwinUnet')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# --- Load Classification Models (Ensemble) ---
# Three models are loaded and their predictions will be averaged (soft voting)
device = torch.device("cpu")
CLF_CHECKPOINT_DIR = os.path.join(BASE_DIR, 'classification', 'checkpoints')

ENSEMBLE_MODELS_INFO = [
    {'name': 'resnet50',            'file': 'resnet50_best.pth'},
    {'name': 'resnest50d',          'file': 'resnest50d_best.pth'},
    {'name': 'tf_efficientnetv2_s', 'file': 'tf_efficientnetv2_s_best.pth'}
]

loaded_classifiers = []
print("Loading classification models...")
for model_info in ENSEMBLE_MODELS_INFO:
    path = os.path.join(CLF_CHECKPOINT_DIR, model_info['file'])
    try:
        model = load_model(model_info['name'], path, device)
        loaded_classifiers.append(model)
        print(f"  [OK] {model_info['name']}")
    except Exception as e:
        print(f"  [FAILED] {model_info['name']}: {e}")

print(f"Classification models loaded: {len(loaded_classifiers)}/{len(ENSEMBLE_MODELS_INFO)}\n")

# --- Load Segmentation Models (YOLOv8) ---
# One YOLO model is loaded per tumor type; stored in a dict keyed by tumor type
seg_models = {}
model_files = {
    'GBM': 'seg_gbm.pt',
    'MG':  'seg_mg.pt',
    'PA':  'seg_pa.pt'
}

print("Loading segmentation models (YOLOv8)...")
for tumor_type, filename in model_files.items():
    model_path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(model_path):
        seg_models[tumor_type] = YOLO(model_path)
        print(f"  [OK] {tumor_type} -> {filename}")
    else:
        print(f"  [MISSING] {tumor_type} -> {filename} not found")

print(f"Segmentation models loaded: {len(seg_models)}/{len(model_files)}\n")


def run_classification(filepath):  # 分類腫瘤種類
    """
    Run ensemble classification on the given image.
    Each loaded model produces a probability distribution over classes;
    the distributions are averaged (soft voting) and the top class is returned.
    Returns: (class_name, confidence_percent, display_label)
    """
    if not loaded_classifiers:
        print("[ERROR] No classifiers available.")
        return "Unknown", 0.0, "No classifiers loaded"

    img_tensor = preprocess_image(filepath)

    # Collect probability distributions from each model
    all_probs = [predict_single_model(model, img_tensor, device) for model in loaded_classifiers]

    # Average the probabilities across all models (soft voting)
    avg_probs = np.mean(all_probs, axis=0)

    pred_idx = np.argmax(avg_probs)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = avg_probs[pred_idx] * 100

    zh_map = {
        "GBM": "膠質母細胞瘤 (GBM)",
        "MG":  "腦膜瘤 (MG)",
        "PA":  "腦下垂體瘤 (PA)",  # 可能會造成潛在錯誤: 將 PT 修改為 PA (Pituitary adenoma)，因此僅修改後面的 value 與 interface 的額外判斷
        "Normal": "正常 (Normal)"
    }

    print(f"[Classification] {os.path.basename(filepath)} -> {pred_class} ({confidence:.2f}%)")
    return pred_class, confidence, zh_map.get(pred_class, pred_class)


def run_unet_inference(filepath, tumor_type, output_path):  # 進行腫瘤分割: 使用 SwinUnet 進行分割 
    """
    Run SwinUnet segmentation for the given tumor type.
    The model directory is expected to contain a checkpoint for the specified type.
    Returns True if the output was saved successfully, False otherwise.
    """
    print(f"[Unet] Running inference for {tumor_type}...")
    try:
        model_dir = os.path.join(BASE_DIR, 'models', 'SwinUnet', tumor_type)
        success = predict_and_save(model_dir, filepath, output_path, label_text=tumor_type)
        if success:
            print(f"[Unet] Segmentation saved to {output_path}")
        else:
            print(f"[Unet] predict_and_save returned False for {tumor_type}")
        return success
    except Exception as e:
        print(f"[Unet] Inference failed for {tumor_type}: {e}")
        return False


def run_segmentation(filepath, tumor_type):  # 進行腫瘤分割
    """
    Attempt segmentation using YOLOv8 first.
    If YOLO is unavailable or produces no mask, fall back to Unet.
    If both fail, the original image is copied to the output path.
    Returns: (output_filename, model_used_label)
    """
    original_filename = os.path.basename(filepath)
    name_without_ext, _ = os.path.splitext(original_filename)
    final_filename = name_without_ext + ".jpg"
    save_path = os.path.join(app.config['RESULT_FOLDER'], final_filename)

    # 先評估 YOLO 是否有對應腫瘤的模型，否則直接使用 Unet
    if tumor_type not in seg_models:
        print(f"[Segmentation] No YOLO model for {tumor_type}, trying SwinUnet...")
        if run_unet_inference(filepath, tumor_type, save_path):
            return final_filename, "SwinUnet"
        return None, "None"
    
    # 使用 YOLO 進行分割  (without auto-saving so we can inspect the mask first)
    print(f"[Segmentation] Running YOLOv8 for {tumor_type}...")
    model = seg_models[tumor_type]
    results = model.predict(source=filepath, save=False, project='static', name='results')

    # Check whether YOLO produced a valid, non-empty mask
    yolo_success = (
        results[0].masks is not None and results[0].masks.data.any()
    )

    if yolo_success:  # 若 YOLO 成功...
        print(f"[Segmentation] YOLOv8 detected lesion for {tumor_type}, saving result...")        
        results[0].names[0] = tumor_type
        results[0].save(filename=save_path)
        return final_filename, "YOLOv8"

    # YOLO 失敗後，若 Unet 成功...
    print(f"[Segmentation] YOLOv8 found no lesion for {tumor_type}, falling back to SwinUnet...")
    if run_unet_inference(filepath, tumor_type, save_path):
        return final_filename, "SwinUnet (YOLO 分割失敗)"

      # YOLO 和 Unet 都失敗後... (回傳原圖)
    print(f"[Segmentation] Both models failed for {tumor_type}, copying original image.")
    shutil.copy(filepath, save_path)
    return final_filename, "無法進行分割，回傳原圖"


# --- Routes ---

@app.route('/download/<filename>')
def download_file(filename):
    """Serve a result file as a download attachment."""
    return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=True)


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main route. On POST: classify the uploaded image, then run segmentation
    if a tumor is detected. On GET: render the upload form.
    """
    if request.method == "POST":
        # 取得當前登入者
        token = request.cookies.get('token')
        if not token: return redirect(url_for('login'))
        try:
            doc_data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            doctor_id = doc_data['doctor_id']
        except:
            return redirect(url_for('login'))
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"\n[Request] Received file: {filename}")

        # Step 1: Classify the image
        pred_class, conf, desc = run_classification(filepath)
        result_filename = None
        model_used = "None"
        model = "無"

        # Step 2: Route to segmentation or skip based on classification result
        if pred_class == "Normal":
            shutil.copy(filepath, os.path.join(app.config['RESULT_FOLDER'], filename))
            result_filename = filename
            print(f"[Request] Result: Normal -- skipping segmentation.")

        elif pred_class in ["GBM", "MG", "PA"]:
            result_filename, model_used = run_segmentation(filepath, tumor_type=pred_class)
            print(f"[Request] Segmentation complete using {model_used}.")

            if model_used == "YOLOv8" or  "Unet" in model_used:  # ... 或是 "SwinUnet" 字串中，包含 "Unet"
                model = model_used
            else:
                model = " 所有模型皆未檢出病灶，顯示原圖"

        return render_template('result.html',
                               original=filename,
                               result=result_filename,
                               label=desc,      # 腫瘤種類的輸出名稱
                               conf=conf,       # 信心程度
                               model=model,   # 使用模型
                               tumor_type=pred_class,  # 腫瘤的英文簡稱
                               is_history=False
                               )
    return render_template("index.html")

        

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve files from the uploads folder (used to display the original image)."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def login_required(f):  # 裝飾器：保護需要登入的頁面
    '''
    什麼是裝飾器? 
    首先， f 的值可能會變成一個函數 (如: history(doctor_id) )；注意到， history() 在一開始被觸發時，沒有得到 id 參數
    接著， decorated(*args, **kwargs) 會被呼叫並執行取得 token 或重新導向的步驟
    最後，將 id 和 原本 f 被呼叫時傳入的參數 (在此範例中， history() 沒有傳入參數) 移併回傳給 f (也就是 history() )
    '''
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('token')
        if not token:
            return redirect(url_for('login'))
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            # 將 doctor_id 傳給後續的 route 函式
            return f(data['doctor_id'], *args, **kwargs)
        except:
            return redirect(url_for('login'))
    return decorated

# 讓所有 HTML 模板都能直接讀取 current_user 變數
@app.context_processor
def inject_user():
    token = request.cookies.get('token')
    if token:
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            return dict(current_user=data)
        except:
            pass
    return dict(current_user=None)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        action = request.form.get('action')
        username = request.form.get('username')
        password = request.form.get('password')
        
        if action == 'register':
            display_name = username
            if not display_name:
                display_name = username  # 防呆機制：若沒填則預設為帳號
            if register_doctor(username, password, display_name):
                return render_template('login.html', msg="註冊成功，請登入！", msg_color="green")
            return render_template('login.html', msg="帳號已存在！", msg_color="red")
            
        elif action == 'login':
            doctor = verify_doctor(username, password)
            if doctor:
                session.pop('last_patient_id', None)
                session.pop('last_patient_name', None)
                # 產生 JWT Token (有效期限 1 天)
                token = jwt.encode({
                    'doctor_id': doctor['id'],
                    'username': doctor['username'],
                    'display_name': doctor['display_name'],
                    'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)    #DOTO-
                }, app.config['SECRET_KEY'], algorithm="HS256")
                
                resp = make_response(redirect(url_for('index')))
                resp.set_cookie('token', token, httponly=True) # 存入 Cookie
                return resp
            return render_template('login.html', msg="帳號或密碼錯誤！", msg_color="red")
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    resp = make_response(redirect(url_for('index')))
    resp.delete_cookie('token')
    return resp

@app.route('/history')
@login_required
def history(doctor_id):
    history_data = get_doctor_history(doctor_id)
    return render_template('history.html', history_data=history_data)

@app.route('/history/photo/<int:photo_id>')
@login_required
def view_history_photo(doctor_id, photo_id):
    # 重複利用 result.html 來顯示歷史結果
    photo = get_photo_by_id(photo_id, doctor_id)
    if not photo:
        return redirect(url_for('history'))
        
    # 將英文代稱轉回中文標籤供畫面顯示
    zh_map = {"GBM": "膠質母細胞瘤", "MG": "腦膜瘤", "PA": "腦下垂體瘤", "Normal": "正常"}
    label = zh_map.get(photo['tumor_type'], photo['tumor_type'])
    
    return render_template('result.html',
                           original=photo['original_image_path'],
                           result=photo['result_image_path'],
                           label=label,
                           conf=photo['confidence'],
                           model=photo['model_used'],
                           tumor_type=photo['tumor_type'],
                           is_history=True,                                         # 標記為歷史模式，前端會隱藏上傳功能
                           photo_id=photo_id,                                     # 傳遞 photo_id
                           current_patient_name=photo['patient_name']) # 顯示目前的患者名稱

@app.route('/setting', methods=['GET', 'POST'])
@login_required
def setting(doctor_id):
    history_data = get_doctor_history(doctor_id) # 供下拉選單刪除病患用
    
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'change_pwd':
            update_doctor_password(doctor_id, request.form.get('new_password'))
        elif action == 'change_name':
            # 更新顯示名稱
            new_name = request.form.get('new_name')
            if new_name:
                update_doctor_display_name(doctor_id, new_name)
                
                # 更新 Cookie 中的 JWT Token，否則前端導覽列不會立刻改變
                token = request.cookies.get('token')
                data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
                data['display_name'] = new_name # 替換成新名字
                
                # 重新簽署 Token
                new_token = jwt.encode(data, app.config['SECRET_KEY'], algorithm="HS256")
                
                # 建立 Response 並設定新 Cookie
                resp = make_response(redirect(url_for('setting')))
                resp.set_cookie('token', new_token, httponly=True)
                return resp
        elif action == 'delete_patient':
            soft_delete_patient(request.form.get('patient_id'), doctor_id)
        elif action == 'delete_account':
            soft_delete_doctor(doctor_id)
            return redirect(url_for('logout'))
        return redirect(url_for('setting'))
        
    return render_template('setting.html', patients=[d['patient'] for d in history_data])

@app.route('/api/search_patient')
@login_required
def api_search_patient(doctor_id):
    """供前端 Modal 搜尋患者名稱的 API"""
    query = request.args.get('q', '')
    patients = search_patients_by_name(doctor_id, query)
    return jsonify(patients)

@app.route('/save_record', methods=['POST'])
@login_required
def save_record(doctor_id):
    """接收 result.html 傳來的存檔要求"""
    patient_name = request.form.get('patient_name')
    if not patient_name:
        return redirect(url_for('index')) # 防呆

    # 取得或新建病患 ID
    patient_id = get_or_create_patient(doctor_id, patient_name)
    
    # 將病患名稱存入 session，這樣下一次開 Modal 就能提示
    session['last_patient_name'] = patient_name

    # 寫入 Photo 資料表
    add_photo(
        original_image=request.form.get('original'),
        result_image=request.form.get('result'),
        tumor_type=request.form.get('tumor_type'),
        confidence=float(request.form.get('conf')),
        model_used=request.form.get('model'),
        patient_id=patient_id
    )
    
    # 存檔完成後，導向該醫師的歷史病歷頁面
    return redirect(url_for('history'))

# ✨ 新增：在歷史紀錄中修改照片所屬患者
@app.route('/reassign_photo', methods=['POST'])
@login_required
def reassign_photo(doctor_id):
    photo_id = request.form.get('photo_id')
    patient_id = request.form.get('patient_id')
    patient_name = request.form.get('patient_name')
    
    if patient_id:
        p_id = int(patient_id)
    else:
        p_id = get_or_create_patient(doctor_id, patient_name)
        
    update_photo_patient(photo_id, p_id, doctor_id)
    return redirect(url_for('history'))

# ✨ 新增：在歷史紀錄中修改病患名稱
@app.route('/rename_patient', methods=['POST'])
@login_required
def rename_patient(doctor_id):
    patient_id = request.form.get('patient_id')
    new_name = request.form.get('new_name')
    if patient_id and new_name:
        update_patient_name(patient_id, doctor_id, new_name)
    return redirect(url_for('history'))
    
if __name__ == "__main__":
    app.run(debug=True, port=5000)