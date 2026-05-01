"""
Backend API — StyleSense NST
"""

import os, sys, time, uuid, json
from flask import Flask, request, jsonify, send_file
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from PIL import Image
import torch
import io
import jwt
from datetime import datetime, timedelta
from functools import wraps

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/extractor"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/nst_optimization"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/nst_fast"))

from optimizer_nst   import run_optimization_nst
from inference       import run_fast_nst
from load_checkpoint import load_extractor

# ── App Init ──────────────────────────────────────────────────
app  = Flask(__name__)
# ── Logging ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# ── CORS (lock to frontend domain) ───────────────────
CORS(app, origins=[
    "http://localhost:3000",
    os.getenv("FRONTEND_URL", "http://localhost:3000")
])

# ── Rate Limiting ─────────────────────────────────────
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)
CORS(app)

UPLOAD_DIR  = os.path.join(os.path.dirname(__file__), "../outputs/uploads")
RESULT_DIR  = os.path.join(os.path.dirname(__file__), "../outputs/api_results")
CHECKPT_DIR = os.path.join(os.path.dirname(__file__), "../checkpoints")
STYLE_IMG   = os.path.join(os.path.dirname(__file__), "../outputs/test_imgs/vangogh_style.jpg")
PRESET_DIR  = os.path.join(os.path.dirname(__file__), "../outputs/test_imgs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ── JWT Auth ───────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "stylesense_t59_secret_dev")
# # PURANA — ye hata do:
# SECRET_KEY = "stylesense_t59_secret"

USERS = {
    "user": {"password": "user123", "role": "user"},
    "dev":  {"password": "dev123",  "role": "developer"},
}

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({"error": "Token missing"}), 401
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            request.user_role = data['role']
        except:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated

def dev_only(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            if data['role'] != 'developer':
                return jsonify({"error": "Developer only"}), 403
            request.user_role = data['role']
        except:
            return jsonify({"error": "Invalid token"}), 401
        return f(*args, **kwargs)
    return decorated

# ── Login ──────────────────────────────────────────────────────
@app.route("/api/login", methods=["POST"])
def login():
    data     = request.get_json()
    username = data.get("username", "")
    password = data.get("password", "")
    user     = USERS.get(username)
    if not user or user["password"] != password:
        return jsonify({"error": "Invalid credentials"}), 401
    token = jwt.encode({
        "username": username,
        "role":     user["role"],
        "exp":      datetime.utcnow() + timedelta(hours=8)
    }, SECRET_KEY, algorithm="HS256")
    return jsonify({"token": token, "role": user["role"], "username": username})

# ── Health ────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    logger.info("Health check called")   # ← ye line add karo
    return jsonify({
        "status" : "ok",
        "gpu"    : torch.cuda.is_available(),
        "device" : torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    })


# ── Stylize ───────────────────────────────────────────────────
@app.route("/api/stylize", methods=["POST"])
@token_required                              # ← ye line add karo
@limiter.limit("10 per minute")             # ← ye line add karo
def stylize():
    logger.info(f"Stylize request - method: {request.form.get('method', 'both')}")  # ← andar pehli line
    """
    Form-data:
      content_image : file  (required)
      style_image   : file  (optional — defaults to Van Gogh)
      method        : str   "fast" | "optimization" | "both"  (default: both)
      img_size      : int   (default: 256)
      iterations    : int   (default: 300, only for optimization)
    Returns JSON with result image paths + metrics
    """
    try:
        method   = request.form.get("method", "both")
        img_size = int(request.form.get("img_size", 128))
        iters    = int(request.form.get("iterations", 300))
        run_id   = str(uuid.uuid4())[:8]

        # Save content image
        if "content_image" not in request.files:
            return jsonify({"error": "content_image required"}), 400
        content_file = request.files["content_image"]
        content_path = os.path.join(UPLOAD_DIR, f"{run_id}_content.jpg")
        Image.open(content_file).convert("RGB").save(content_path)

        # Save style image (or use default)
        if "style_image" in request.files:
            style_file = request.files["style_image"]
            style_path = os.path.join(UPLOAD_DIR, f"{run_id}_style.jpg")
            Image.open(style_file).convert("RGB").save(style_path)
        else:
            style_path = STYLE_IMG

        results = {"run_id": run_id, "method": method}

        # Fast NST
        if method in ("fast", "both"):
            fast_out = os.path.join(RESULT_DIR, f"{run_id}_fast.jpg")
            t0 = time.time()
            fast_metrics = run_fast_nst(
                content_path   = content_path,
                style_path     = style_path,
                checkpoint     = os.path.join(CHECKPT_DIR, "fast_nst_epoch2.pth"),
                extractor_ckpt = os.path.join(CHECKPT_DIR, "best_extractor.pth"),
                output_path    = fast_out,
                img_size       = img_size,
            )
            results["fast"] = {
                **fast_metrics,
                "result_url": f"/api/result/{run_id}_fast.jpg"
            }

        # Optimization NST
        if method in ("optimization", "both"):
            opt_out = os.path.join(RESULT_DIR, f"{run_id}_opt.jpg")
            opt_metrics = run_optimization_nst(
                content_path = content_path,
                style_path   = style_path,
                checkpoint   = os.path.join(CHECKPT_DIR, "best_extractor.pth"),
                output_path  = opt_out,
                img_size     = img_size,
                iterations   = iters,
                save_every   = iters,
            )
            results["optimization"] = {
                **opt_metrics,
                "result_url": f"/api/result/{run_id}_opt.jpg"
            }

        # Speedup
        if method == "both":
            fast_ms = results["fast"]["runtime_ms"]
            opt_ms  = results["optimization"]["runtime_ms"]
            results["speedup"] = round(opt_ms / max(fast_ms, 0.1), 1)

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Preset Styles ──────────────────────────────────────────
PRESET_DIR = os.path.join(os.path.dirname(__file__), "../outputs/test_imgs")

@app.route("/api/styles", methods=["GET"])
def list_styles():
    presets = [
        {"id": "vangogh",  "name": "Van Gogh — Starry Night",     "file": "vangogh_style.jpg"},
        {"id": "ghibli",   "name": "Ghibli — Studio Magic",       "file": "ghibl.jpg"},
        {"id": "monalisa", "name": "Da Vinci — Mona Lisa",        "file": "Mona_Lisa.jpg"},
        {"id": "abstract", "name": "Abstract",                    "file": "style.jpg"},
        {"id": "vgtotoro", "name": "Van Gogh x Totoro",           "file": "Vincent van Gogh and My Neighbor Totoro.jpg"},
    ]
    return jsonify({"presets": presets})

@app.route("/api/styles/<style_id>", methods=["GET"])
def get_style_image(style_id):
    presets = {
        "vangogh":  "vangogh_style.jpg",
        "ghibli":   "ghibl.jpg",
        "monalisa": "Mona_Lisa.jpg",
        "abstract": "style.jpg",
        "vgtotoro": "Vincent van Gogh and My Neighbor Totoro.jpg",
    }
    if style_id not in presets:
        return jsonify({"error": "Not found"}), 404
    return send_file(os.path.join(PRESET_DIR, presets[style_id]), mimetype="image/jpeg")


# ── Serve Result Image ────────────────────────────────────────
@app.route("/api/result/<filename>", methods=["GET"])
def get_result(filename):
    path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    return send_file(path, mimetype="image/jpeg")


# ── Benchmark ─────────────────────────────────────────────────
@app.route("/api/benchmark", methods=["POST"])
def benchmark():
    """
    Runs both NST on provided image at multiple resolutions.
    Form-data:
      content_image : file
      style_image   : file (optional)
      resolutions   : str  "128,256" (default)
      iterations    : int  (default: 100)
    """
    try:
        resolutions = [int(r) for r in
                       request.form.get("resolutions", "128,256").split(",")]
        iters   = int(request.form.get("iterations", 100))
        run_id  = str(uuid.uuid4())[:8]
        results = []

        content_file = request.files["content_image"]
        content_path = os.path.join(UPLOAD_DIR, f"{run_id}_content.jpg")
        Image.open(content_file).convert("RGB").save(content_path)

        style_path = STYLE_IMG
        if "style_image" in request.files:
            style_path = os.path.join(UPLOAD_DIR, f"{run_id}_style.jpg")
            Image.open(request.files["style_image"]).convert("RGB").save(style_path)

        for res in resolutions:
            # Opt NST
            opt_out = os.path.join(RESULT_DIR, f"{run_id}_opt_{res}.jpg")
            opt_m   = run_optimization_nst(
                content_path=content_path, style_path=style_path,
                checkpoint=os.path.join(CHECKPT_DIR, "best_extractor.pth"),
                output_path=opt_out, img_size=res,
                iterations=iters, save_every=iters,
            )
            # Fast NST
            fast_out = os.path.join(RESULT_DIR, f"{run_id}_fast_{res}.jpg")
            fast_m   = run_fast_nst(
                content_path=content_path, style_path=style_path,
                checkpoint=os.path.join(CHECKPT_DIR, "fast_nst_epoch2.pth"),
                extractor_ckpt=os.path.join(CHECKPT_DIR, "best_extractor.pth"),
                output_path=fast_out, img_size=res,
            )
            results.append({
                "resolution"    : res,
                "opt_runtime_ms": opt_m["runtime_ms"],
                "fast_runtime_ms": fast_m["runtime_ms"],
                "speedup"       : round(opt_m["runtime_ms"] / max(fast_m["runtime_ms"], 0.1), 1),
                "opt_style_loss": opt_m["style_loss"],
                "fast_style_loss": fast_m["style_loss"],
            })

        return jsonify({"run_id": run_id, "benchmark": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Recommend ─────────────────────────────────────────────────
@app.route("/api/recommend", methods=["GET"])
def recommend():
    """
    Query params:
      use_case    : "realtime" | "quality" | "batch"
      max_time_ms : int (max acceptable runtime)
      quality     : "high" | "medium" | "low"
    """
    use_case    = request.args.get("use_case", "realtime")
    max_time_ms = int(request.args.get("max_time_ms", 500))
    quality     = request.args.get("quality", "medium")

    # Rule-based recommendation engine
    if use_case == "realtime" or max_time_ms < 500:
        method = "fast"
        reason = "Fast NST delivers results in ~160ms — ideal for real-time filters and mobile apps."
        tradeoff = "Slightly lower style fidelity vs optimization NST."

    elif use_case == "quality" or quality == "high":
        method = "optimization"
        reason = "Optimization NST produces highest quality by iteratively refining the image."
        tradeoff = "Slower (~3s for 300 iterations at 256x256)."

    elif use_case == "batch":
        method = "fast"
        reason = "Fast NST processes large batches efficiently with consistent quality."
        tradeoff = "Style is fixed to trained style; less flexible than optimization NST."

    else:
        method = "fast"
        reason = "Fast NST is recommended as a balanced default for most use cases."
        tradeoff = "For maximum quality, switch to optimization NST."

    return jsonify({
        "recommended_method" : method,
        "reason"             : reason,
        "tradeoff"           : tradeoff,
        "benchmarks"         : {
            "fast_avg_ms"    : 160,
            "opt_avg_ms"     : 3086,
            "speedup"        : "~19x"
        }
    })


if __name__ == "__main__":
    print("\n  StyleSense Backend API")
    print("  GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("  Starting server on http://localhost:5000\n")
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))

# Shortcut — bina /api prefix ke bhi kaam kare
from flask import redirect
@app.route('/health', methods=['GET'])
@app.route('/api/health', methods=['GET'])
def health_shortcut():
    return health()
