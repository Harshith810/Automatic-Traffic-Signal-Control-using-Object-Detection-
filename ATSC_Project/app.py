# atcs_app.py
from flask import Flask, render_template, request, url_for, send_from_directory
from ultralytics import YOLO
import os, cv2, uuid, math, shutil
import numpy as np

# ==================== CONFIG =====================
# update these paths if your models are elsewhere
NORMAL_MODEL_PATH = r"C:\Users\harsh\OneDrive\Desktop\ATSC Project\models\normal_model\best.pt"
EMERGENCY_MODEL_PATH = r"C:\Users\harsh\OneDrive\Desktop\ATSC Project\models\emergency_model\best2.pt"

CONF_NORMAL = 0.35
CONF_EMERGENCY = 0.35

# label canonicalization sets
VEHICLE_CLASSES = {"car","bus","truck","bike","bicycle","motorbike","motorcycle","auto","rickshaw"}
EMERGENCY_CLASSES = {"ambulance","fire_truck","firetruck","police_car","police"}

# signal timing (simple formula)
BASE_GREEN = 8
PER_VEHICLE = 1.5
MIN_GREEN = 6
MAX_GREEN = 25
YELLOW_SEC = 2.0

# lane split for single-image lane logic (unused in two-camera mode but kept)
SPLIT_L = 0.34
SPLIT_R = 0.68

# ==================== FLASK SETUP =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")
RESULT_DIR = os.path.join(STATIC_DIR, "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
app.jinja_env.cache = {}

# ==================== LOAD MODELS =====================
print("Loading models (this may take a moment)...")
normal_model = YOLO(NORMAL_MODEL_PATH)
emg_model = YOLO(EMERGENCY_MODEL_PATH)
print("Models loaded.")

# ==================== HELPERS =====================
def iou_xyxy(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    areaB = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    denom = areaA + areaB - inter + 1e-9
    return inter / denom

def nms_ensemble(dets, iou_thr=0.5):
    grouped = {}
    for box, lbl, conf in dets:
        grouped.setdefault(lbl, []).append((box, lbl, conf))
    kept = []
    for lbl, items in grouped.items():
        items.sort(key=lambda x: x[2], reverse=True)
        used = [False]*len(items)
        for i,(bi,li,pi) in enumerate(items):
            if used[i]: continue
            kept.append((bi,li,pi))
            for j in range(i+1,len(items)):
                if used[j]: continue
                bj,lj,pj = items[j]
                if iou_xyxy(bi,bj) > iou_thr:
                    used[j] = True
    return kept

def compute_green(count):
    sec = BASE_GREEN + PER_VEHICLE * count
    return int(max(MIN_GREEN, min(MAX_GREEN, sec)))

def annotate_image_with_dets(img, dets):
    """Draw boxes and return annotated image and counts + emergency flag."""
    h, w = img.shape[:2]
    class_counts = {}
    emergency_present = False

    for (x1,y1,x2,y2), label, conf in dets:
        x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
        # ignore tiny boxes
        if (x2-x1)*(y2-y1) < 500: 
            continue
        color = (80,220,80) if label in VEHICLE_CLASSES else (0,215,255) if label in EMERGENCY_CLASSES else (180,180,180)
        if label in EMERGENCY_CLASSES:
            emergency_present = True
        class_counts[label] = class_counts.get(label,0) + 1
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, f"{label} {conf:.2f}", (x1, max(20,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return img, class_counts, emergency_present

def run_ensemble_on_image(image_bgr):
    """Run both models on a single frame (numpy BGR) and return merged detections after nms."""
    dets = []
    # normal model
    try:
        r1 = normal_model.predict(image_bgr, conf=CONF_NORMAL, verbose=False)
        if len(r1) and r1[0].boxes is not None:
            names = normal_model.names
            for b, c, p in zip(r1[0].boxes.xyxy.cpu().numpy(),
                               r1[0].boxes.cls.cpu().numpy(),
                               r1[0].boxes.conf.cpu().numpy()):
                dets.append((b, names[int(c)].lower(), float(p)))
    except Exception as e:
        print("Normal model predict error:", e)

    # emergency model
    try:
        r2 = emg_model.predict(image_bgr, conf=CONF_EMERGENCY, verbose=False)
        if len(r2) and r2[0].boxes is not None:
            names = emg_model.names
            for b, c, p in zip(r2[0].boxes.xyxy.cpu().numpy(),
                               r2[0].boxes.cls.cpu().numpy(),
                               r2[0].boxes.conf.cpu().numpy()):
                dets.append((b, names[int(c)].lower(), float(p)))
    except Exception as e:
        print("Emergency model predict error:", e)

    # merge via simple per-class NMS
    return nms_ensemble(dets, iou_thr=0.45)

def process_image_file(filepath, out_name_prefix):
    img = cv2.imread(filepath)
    if img is None:
        raise RuntimeError("Failed to read image: " + filepath)
    dets = run_ensemble_on_image(img.copy())
    ann_img, counts, emergency_present = annotate_image_with_dets(img.copy(), dets)
    outname = f"{out_name_prefix}_{uuid.uuid4().hex[:8]}.jpg"
    outpath = os.path.join(RESULT_DIR, outname)
    cv2.imwrite(outpath, ann_img)
    total_count = sum(counts.values())
    return outname, counts, total_count, emergency_present

def process_video_file(filepath, out_name_prefix, sample_frame_step=3):
    """Read video, annotate frames and write to output video. 
       Returns output filename, average_count, emergency_present flag.
       sample_frame_step: process every n-th frame (speed vs accuracy).
    """
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video: " + filepath)

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outname = f"{out_name_prefix}_{uuid.uuid4().hex[:8]}.mp4"
    outpath = os.path.join(RESULT_DIR, outname)
    writer = cv2.VideoWriter(outpath, fourcc, fps, (width, height))

    total_counts = []
    emergency_present = False
    frame_idx = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_frame_step != 0:
            # still write the frame unchanged to keep video length consistent
            writer.write(frame)
            continue

        dets = run_ensemble_on_image(frame.copy())
        ann_frame, counts, emg = annotate_image_with_dets(frame.copy(), dets)
        # write annotated frame
        writer.write(ann_frame)
        total_counts.append(sum(counts.values()))
        if emg:
            emergency_present = True
        processed_frames += 1

    cap.release()
    writer.release()

    avg_count = int(np.mean(total_counts)) if processed_frames>0 else 0
    return outname, avg_count, emergency_present

# ==================== ROUTES =====================
@app.route("/")
def home():
    return render_template("index2.html")

@app.route("/upload_two", methods=["POST"])
def upload_two():
    # Expecting two files: 'laneA' and 'laneB'
    fA = request.files.get("laneA")
    fB = request.files.get("laneB")
    if not fA or not fB:
        return render_template("index2.html", error="Please upload both Lane A and Lane B files (image or video).")

    # Save uploads
    fnameA = uuid.uuid4().hex + "_" + secure_filename_os(fA.filename)
    fnameB = uuid.uuid4().hex + "_" + secure_filename_os(fB.filename)
    pathA = os.path.join(UPLOAD_DIR, fnameA)
    pathB = os.path.join(UPLOAD_DIR, fnameB)
    fA.save(pathA)
    fB.save(pathB)

    # Decide per-file processing branch (image / video)
    out_info = {}
    for lane_key, path in [("A", pathA), ("B", pathB)]:
        ext = os.path.splitext(path)[1].lower()
        prefix = f"lane{lane_key}"
        try:
            if ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                outname, counts, total_count, emergency = process_image_file(path, prefix)
                out_info[lane_key] = {
                    "type": "image",
                    "input_path": url_for('static', filename=f"uploads/{os.path.basename(path)}"),
                    "output_path": url_for('static', filename=f"results/{outname}"),
                    "counts": counts,
                    "total": total_count,
                    "emergency": emergency
                }
            elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
                outname, avg_count, emergency = process_video_file(path, prefix, sample_frame_step=4)
                out_info[lane_key] = {
                    "type": "video",
                    "input_path": url_for('static', filename=f"uploads/{os.path.basename(path)}"),
                    "output_path": url_for('static', filename=f"results/{outname}"),
                    "counts": None,
                    "total": avg_count,
                    "emergency": emergency
                }
            else:
                # unknown type
                return render_template("index2.html", error=f"Unsupported file type for {os.path.basename(path)}")
        except Exception as e:
            return render_template("index2.html", error=f"Processing error for {os.path.basename(path)}: {e}")

    # Decision logic: emergency overrides density
    A_info = out_info["A"]
    B_info = out_info["B"]
    decision_text = ""
    green_duration = {}

    if A_info["emergency"] and not B_info["emergency"]:
        winner = "Lane A (Emergency)"
        decision_text = "Emergency detected in Lane A → priority green."
    elif B_info["emergency"] and not A_info["emergency"]:
        winner = "Lane B (Emergency)"
        decision_text = "Emergency detected in Lane B → priority green."
    elif A_info["emergency"] and B_info["emergency"]:
        # both have emergency: choose the one with earlier higher confidence? we use totals
        winner = "Lane A" if A_info["total"] >= B_info["total"] else "Lane B"
        decision_text = "Emergency detected in both lanes — giving green to higher density lane."
    else:
        # no emergencies -> compare totals
        if A_info["total"] > B_info["total"]:
            winner = "Lane A"
            decision_text = f"Lane A has higher traffic ({A_info['total']} vs {B_info['total']})."
        elif B_info["total"] > A_info["total"]:
            winner = "Lane B"
            decision_text = f"Lane B has higher traffic ({B_info['total']} vs {A_info['total']})."
        else:
            winner = "Lane A"  # tie-breaker
            decision_text = f"Tie ({A_info['total']} each). Defaulting to Lane A."

    # compute green durations (simple)
    green_duration[winner] = compute_green(A_info["total"] if winner.startswith("Lane A") else B_info["total"])

    # summary strings for template
    summary = {
        "laneA": {
            "total": A_info["total"],
            "emergency": A_info["emergency"],
            "output": A_info["output_path"],
            "input": A_info["input_path"],
            "type": A_info["type"],
            "counts": A_info["counts"]
        },
        "laneB": {
            "total": B_info["total"],
            "emergency": B_info["emergency"],
            "output": B_info["output_path"],
            "input": B_info["input_path"],
            "type": B_info["type"],
            "counts": B_info["counts"]
        },
        "decision": decision_text,
        "winner": winner,
        "green_duration": green_duration.get(winner, compute_green(max(A_info["total"], B_info["total"])))
    }

    return render_template("index2.html", result=summary)

@app.route('/static/<path:path>')
def static_send(path):
    return send_from_directory('static', path)

# small helper to sanitize filenames (very simple)
def secure_filename_os(fn):
    return "".join(c for c in fn if c.isalnum() or c in "._- ").strip().replace(" ", "_")

if __name__ == "__main__":
    print("Template folder:", app.template_folder)
    print("Static folder:", app.static_folder)
    print("Uploads:", UPLOAD_DIR)
    print("Results:", RESULT_DIR)
    app.run(debug=True)




