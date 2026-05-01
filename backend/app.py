import os
import sys
import time
import uuid
import logging
from pathlib import Path
from datetime import datetime, timedelta, timezone
from functools import wraps

import bcrypt
import jwt
import torch
from PIL import Image
from flask import Flask, jsonify, request, send_file, send_from_directory, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

sys.path.append(str(PROJECT_ROOT / "src" / "nst" / "fast"))
sys.path.append(str(PROJECT_ROOT / "src" / "nst" / "optimization"))
sys.path.append(str(PROJECT_ROOT / "src" / "extractor"))

# Change only these imports if your real function names differ
from inference import run_fast_nst
from optimizer_nst import run_optimization_nst

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "https://t59-comparative-analysis-of-optimization-based-vs-63hcybggz.vercel.app"
)
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "stylesense-t59-dev-secret")
PORT = int(os.getenv("PORT", 5000))

app.config["SECRET_KEY"] = JWT_SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

CORS(
    app,
    resources={r"/api/*": {"origins": [FRONTEND_URL, "http://localhost:3000"]}},
    supports_credentials=True
)

Talisman(
    app,
    force_https=False,
    content_security_policy=None
)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

UPLOAD_DIR = PROJECT_ROOT / "outputs" / "uploads"
RESULT_DIR = PROJECT_ROOT / "outputs" / "api_results"
STYLE_DIR = PROJECT_ROOT / "outputs" / "test_imgs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FAST_CKPT = CHECKPOINT_DIR / "nst_epoch2.pth"
EXTRACTOR_CKPT = CHECKPOINT_DIR / "best_extractor.pth"

STYLE_PRESETS = [
    {"id": "vangogh", "name": "Van Gogh", "filename": "vangogh_style.jpg"},
    {"id": "ghibli", "name": "Ghibli", "filename": "ghibli_style.jpg"},
    {"id": "monalisa", "name": "Mona Lisa", "filename": "monalisa_style.jpg"},
    {"id": "abstract", "name": "Abstract", "filename": "abstract_style.jpg"},
    {"id": "vangogh_totoro", "name": "Van Gogh x Totoro", "filename": "vangogh_totoro_style.jpg"},
]
STYLE_INDEX = {item["id"]: item for item in STYLE_PRESETS}


def json_error(message, status=400):
    return jsonify({"error": message}), status


def device_info():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def hash_from_env(env_name, default_plain):
    raw = os.getenv(env_name, default_plain).encode("utf-8")
    return bcrypt.hashpw(raw, bcrypt.gensalt())


USERS = {
    "user": {
        "role": "user",
        "password_hash": hash_from_env("DEMO_USER_PASSWORD", "user123"),
    },
    "dev": {
        "role": "developer",
        "password_hash": hash_from_env("DEMO_DEV_PASSWORD", "dev123"),
    },
}


def create_token(username, role):
    payload = {
        "sub": username,
        "role": role,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=8),
    }
    return jwt.encode(payload, app.config["SECRET_KEY"], algorithm="HS256")


def token_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return json_error("Missing or invalid Authorization header", 401)

        token = auth_header.split(" ", 1)[1].strip()
        try:
            payload = jwt.decode(token, app.config["SECRET_KEY"], algorithms=["HS256"])
            g.user = {
                "username": payload.get("sub"),
                "role": payload.get("role"),
            }
        except jwt.ExpiredSignatureError:
            return json_error("Token expired", 401)
        except jwt.InvalidTokenError:
            return json_error("Invalid token", 401)

        return fn(*args, **kwargs)

    return wrapper


def save_rgb_image(file_storage, out_path: Path):
    image = Image.open(file_storage.stream).convert("RGB")
    image.save(out_path)
    return out_path


def resolve_style_path(style_id: str) -> Path:
    meta = STYLE_INDEX.get(style_id)
    if not meta:
        raise ValueError(f"Invalid style id: {style_id}")

    style_path = STYLE_DIR / meta["filename"]
    if not style_path.exists():
        raise FileNotFoundError(f"Style file not found for {style_id}: {style_path}")

    return style_path


def run_fast_wrapper(content_path: Path, style_path: Path, output_path: Path, img_size: int):
    t0 = time.time()

    meta = run_fast_nst(
        content_path=str(content_path),
        style_path=str(style_path),
        checkpoint=str(FAST_CKPT),
        extractor_ckpt=str(EXTRACTOR_CKPT),
        output_path=str(output_path),
        imgsize=img_size,
    )

    if not output_path.exists():
        raise RuntimeError("Fast NST did not produce an output image")

    elapsed = round(time.time() - t0, 4)
    meta = meta if isinstance(meta, dict) else {}

    if "runtime_ms" in meta:
        elapsed = round(float(meta["runtime_ms"]) / 1000.0, 4)

    return {
        "method": "fast",
        "output_url": f"/api/result/{output_path.name}",
        "time_seconds": elapsed,
        "metrics": meta,
    }


def run_opt_wrapper(content_path: Path, style_path: Path, output_path: Path, img_size: int, iterations: int):
    t0 = time.time()

    meta = run_optimization_nst(
        content_path=str(content_path),
        style_path=str(style_path),
        checkpoint=str(EXTRACTOR_CKPT),
        output_path=str(output_path),
        imgsize=img_size,
        iterations=iterations,
        save_every=iterations,
    )

    if not output_path.exists():
        raise RuntimeError("Optimization NST did not produce an output image")

    elapsed = round(time.time() - t0, 4)
    meta = meta if isinstance(meta, dict) else {}

    if "runtime_ms" in meta:
        elapsed = round(float(meta["runtime_ms"]) / 1000.0, 4)

    return {
        "method": "optimization",
        "output_url": f"/api/result/{output_path.name}",
        "time_seconds": elapsed,
        "metrics": meta,
    }


@app.before_request
def log_request():
    logger.info("REQ %s %s ip=%s", request.method, request.path, request.remote_addr)


@app.after_request
def log_response(response):
    logger.info("RES %s %s status=%s", request.method, request.path, response.status_code)
    return response


@app.route("/api/health", methods=["GET"])
def health():
    try:
        return jsonify({
            "status": "ok",
            "gpu": torch.cuda.is_available(),
            "device": device_info()
        }), 200
    except Exception as e:
        logger.exception("health failed: %s", e)
        return json_error("Internal server error", 500)


@app.route("/api/login", methods=["POST"])
@limiter.limit("10 per minute")
def login():
    try:
        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""

        if not username or not password:
            return json_error("username and password are required", 400)

        record = USERS.get(username)
        if not record:
            return json_error("invalid credentials", 401)

        if not bcrypt.checkpw(password.encode("utf-8"), record["password_hash"]):
            return json_error("invalid credentials", 401)

        token = create_token(username, record["role"])

        return jsonify({
            "message": "login successful",
            "token": token,
            "user": {
                "username": username,
                "role": record["role"]
            }
        }), 200

    except Exception as e:
        logger.exception("login failed: %s", e)
        return json_error("Internal server error", 500)


@app.route("/api/styles", methods=["GET"])
def list_styles():
    try:
        return jsonify({
            "presets": [{"id": s["id"], "name": s["name"]} for s in STYLE_PRESETS]
        }), 200
    except Exception as e:
        logger.exception("list_styles failed: %s", e)
        return json_error("Internal server error", 500)


@app.route("/api/styles/<style_id>", methods=["GET"])
def get_style(style_id):
    try:
        style_path = resolve_style_path(style_id)
        return send_file(style_path)
    except ValueError as e:
        return json_error(str(e), 404)
    except FileNotFoundError as e:
        logger.warning("style file missing: %s", e)
        return json_error(str(e), 404)
    except Exception as e:
        logger.exception("get_style failed: %s", e)
        return json_error("Internal server error", 500)


@app.route("/api/result/<filename>", methods=["GET"])
def get_result(filename):
    try:
        return send_from_directory(RESULT_DIR, filename)
    except Exception as e:
        logger.exception("get_result failed: %s", e)
        return json_error("File not found", 404)


@app.route("/api/recommend", methods=["GET"])
def recommend():
    try:
        scenario = (request.args.get("usecase") or "realtime").strip().lower()

        rules = {
            "realtime": {
                "recommended_method": "realtime-fast",
                "reason": "Fast NST is best for real-time previews and interactive use.",
            },
            "real-time": {
                "recommended_method": "realtime-fast",
                "reason": "Fast NST is best for real-time previews and interactive use.",
            },
            "quality": {
                "recommended_method": "quality-opt",
                "reason": "Optimization NST is better when output quality matters more than speed.",
            },
            "quality-first": {
                "recommended_method": "quality-opt",
                "reason": "Optimization NST is better when output quality matters more than speed.",
            },
            "batch": {
                "recommended_method": "realtime-fast",
                "reason": "Fast NST scales better for many images.",
            },
        }

        chosen = rules.get(scenario, rules["realtime"])
        return jsonify({
            "scenario": scenario,
            **chosen
        }), 200

    except Exception as e:
        logger.exception("recommend failed: %s", e)
        return json_error("Internal server error", 500)


@app.route("/api/stylize", methods=["POST"])
@token_required
@limiter.limit("10 per minute")
def stylize():
    try:
        method = (request.form.get("method") or "fast").strip().lower()
        img_size = int(request.form.get("imgsize", 512))
        iterations = int(request.form.get("iterations", 300))
        style_id = request.form.get("style_id")

        if method not in {"fast", "optimization", "both"}:
            return json_error("method must be fast, optimization, or both", 422)

        content_file = request.files.get("content_image") or request.files.get("content")
        if not content_file:
            return json_error("content image is required", 400)

        run_id = uuid.uuid4().hex[:12]
        content_path = UPLOAD_DIR / f"{run_id}_content.jpg"
        save_rgb_image(content_file, content_path)

        if request.files.get("style_image"):
            style_path = UPLOAD_DIR / f"{run_id}_style.jpg"
            save_rgb_image(request.files["style_image"], style_path)
        elif style_id:
            style_path = resolve_style_path(style_id)
        else:
            return json_error("style image or style_id is required", 400)

        logger.info(
            "stylize user=%s method=%s iterations=%s img_size=%s style_id=%s",
            g.user["username"], method, iterations, img_size, style_id
        )

        response = {"run_id": run_id, "method": method}

        if method in {"fast", "both"}:
            fast_output = RESULT_DIR / f"{run_id}_fast.jpg"
            fast_result = run_fast_wrapper(content_path, style_path, fast_output, img_size)
            response["fast"] = fast_result

        if method in {"optimization", "both"}:
            opt_output = RESULT_DIR / f"{run_id}_opt.jpg"
            opt_result = run_opt_wrapper(content_path, style_path, opt_output, img_size, iterations)
            response["optimization"] = opt_result

        if method == "fast":
            return jsonify(response["fast"]), 200

        if method == "optimization":
            return jsonify(response["optimization"]), 200

        fast_time = response["fast"]["time_seconds"]
        opt_time = response["optimization"]["time_seconds"]
        response["speedup"] = round(opt_time / max(fast_time, 0.001), 2)

        return jsonify(response), 200

    except ValueError as e:
        logger.warning("stylize validation error: %s", e)
        return json_error(str(e), 422)
    except FileNotFoundError as e:
        logger.warning("stylize file error: %s", e)
        return json_error(str(e), 404)
    except Exception as e:
        logger.exception("stylize failed: %s", e)
        return json_error("Internal server error", 500)


@app.route("/api/benchmark", methods=["POST"])
@token_required
@limiter.limit("5 per minute")
def benchmark():
    try:
        img_size = int(request.form.get("imgsize", 512))
        iterations = int(request.form.get("iterations", 300))
        style_id = request.form.get("style_id")

        content_file = request.files.get("content_image") or request.files.get("content")
        if not content_file:
            return json_error("content image is required", 400)

        run_id = uuid.uuid4().hex[:12]
        content_path = UPLOAD_DIR / f"{run_id}_bench_content.jpg"
        save_rgb_image(content_file, content_path)

        if request.files.get("style_image"):
            style_path = UPLOAD_DIR / f"{run_id}_bench_style.jpg"
            save_rgb_image(request.files["style_image"], style_path)
        elif style_id:
            style_path = resolve_style_path(style_id)
        else:
            return json_error("style image or style_id is required", 400)

        fast_output = RESULT_DIR / f"{run_id}_bench_fast.jpg"
        opt_output = RESULT_DIR / f"{run_id}_bench_opt.jpg"

        fast_result = run_fast_wrapper(content_path, style_path, fast_output, img_size)
        opt_result = run_opt_wrapper(content_path, style_path, opt_output, img_size, iterations)

        fast_metrics = fast_result.get("metrics", {})
        opt_metrics = opt_result.get("metrics", {})

        payload = {
            "run_id": run_id,
            "fast_time_seconds": fast_result["time_seconds"],
            "optimization_time_seconds": opt_result["time_seconds"],
            "speedup": round(
                opt_result["time_seconds"] / max(fast_result["time_seconds"], 0.001),
                2
            ),
            "fast_loss": fast_metrics.get("loss") or fast_metrics.get("final_loss"),
            "optimization_loss": opt_metrics.get("loss") or opt_metrics.get("final_loss"),
            "fast_output_url": fast_result["output_url"],
            "optimization_output_url": opt_result["output_url"],
        }

        return jsonify(payload), 200

    except ValueError as e:
        logger.warning("benchmark validation error: %s", e)
        return json_error(str(e), 422)
    except FileNotFoundError as e:
        logger.warning("benchmark file error: %s", e)
        return json_error(str(e), 404)
    except Exception as e:
        logger.exception("benchmark failed: %s", e)
        return json_error("Internal server error", 500)


if __name__ == "__main__":
    logger.info("Starting StyleSense backend on port %s", PORT)
    app.run(host="0.0.0.0", port=PORT, debug=False)