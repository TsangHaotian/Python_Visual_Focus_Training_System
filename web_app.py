"""
Flask 网页：摄像头 MJPEG + 专注度 API + 前端游戏/图表。
模型路径默认固定为项目内 `pt_model/model-ui.pt`（与工作目录无关）。
"""
from __future__ import annotations

import argparse
import math
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from ai_strategy import AIStrategyManager

app = Flask(__name__, static_folder="static", template_folder="templates")

_state_lock = threading.Lock()
_state: dict[str, Any] = {
    "engine_ok": False,
    "load_error": None,
    "face_detected": False,
    "cls_idx": None,
    "conf": 0.0,
    "probs": [0.0, 0.0, 0.0, 0.0],
}

# AI策略管理器
_ai_manager = AIStrategyManager("sk-f8b590c5a56e44b5be79ceac7dd5afda")

_latest_jpeg: bytes | None = None
_jpeg_lock = threading.Lock()

_stop = threading.Event()
_cam_thread: threading.Thread | None = None


def project_root() -> Path:
    return Path(__file__).resolve().parent


def default_model_path() -> Path:
    return project_root() / "rule_model" / "models" / "face_landmarker.task"


def _focus_soft_0_3(probs: list[float], face_detected: bool) -> float:
    """与 UI/游戏一致：期望等级 ∑(i·p_i)，范围 clamp 到 [0,3]；无人脸为 0。"""
    if not face_detected:
        return 0.0
    p4 = [float(x) for x in list(probs)[:4]]
    while len(p4) < 4:
        p4.append(0.0)
    s = sum(i * p4[i] for i in range(4))
    return max(0.0, min(3.0, float(s)))


def _obstacle_level_0_3(cls_idx: int | None, face_detected: bool) -> float:
    """障碍高度等级与专注度同一 0–3 标尺：专注越低障碍越高（3−cls）。"""
    if not face_detected or cls_idx is None:
        return 3.0
    if not (0 <= cls_idx <= 3):
        return 1.5
    return float(3 - cls_idx)


def _update_state_from_result(result: dict[str, Any]) -> None:
    with _state_lock:
        _state["face_detected"] = bool(result.get("face_detected"))
        _state["cls_idx"] = result.get("cls_idx")
        _state["conf"] = float(result.get("conf", 0.0))
        probs = result.get("probs") or [0.0, 0.0, 0.0, 0.0]
        p4 = [float(x) for x in list(probs)[:4]]
        while len(p4) < 4:
            p4.append(0.0)
        _state["probs"] = p4
        
        # 与摄像头标注、徽章、曲线、小球高度统一为 0–3（软标签为期望等级）
        fd = bool(result.get("face_detected"))
        precise_focus = _focus_soft_0_3(p4, fd)
        _ai_manager.add_focus_data(precise_focus)


def _draw_error_frame(msg: str, w: int = 640, h: int = 480) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    y = 36
    for line in _wrap_text(msg, 70):
        cv2.putText(
            img,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (60, 60, 255),
            2,
            cv2.LINE_AA,
        )
        y += 26
    return img


def _wrap_text(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    while text:
        lines.append(text[:max_chars])
        text = text[max_chars:]
    return lines or [""]


def _resize_for_stream(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _resize_capture(frame: np.ndarray, max_width: int) -> np.ndarray:
    """先缩小采集画面再做人脸与推理，显著减轻 CPU。"""
    if max_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _euler_yaw_pitch_roll_from_4x4(mat) -> tuple[float, float, float]:
    m = np.asarray(mat, dtype=np.float32)
    r = m[:3, :3]
    forward = r @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
    up = r @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
    yaw = math.atan2(float(forward[0]), float(-forward[2]))
    pitch = math.atan2(float(forward[1]), math.sqrt(float(forward[0] ** 2 + forward[2] ** 2)))
    roll = math.atan2(float(up[0]), float(up[1]))
    return (math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def _down_proxy(face_landmarks) -> float | None:
    try:
        left_eye = face_landmarks[33]
        right_eye = face_landmarks[263]
        chin = face_landmarks[152]
        nose = face_landmarks[1]
    except Exception:
        return None
    eye_y = 0.5 * (float(left_eye.y) + float(right_eye.y))
    chin_y = float(chin.y)
    nose_y = float(nose.y)
    denom = chin_y - eye_y
    if denom <= 1e-6:
        return None
    return (nose_y - eye_y) / denom


def _ear(face_landmarks, a: int, b: int, c: int, d: int, e: int, f: int) -> float | None:
    try:
        pa = face_landmarks[a]
        pb = face_landmarks[b]
        pc = face_landmarks[c]
        pd = face_landmarks[d]
        pe = face_landmarks[e]
        pf = face_landmarks[f]
    except Exception:
        return None
    ax, ay = float(pa.x), float(pa.y)
    bx, by = float(pb.x), float(pb.y)
    cx, cy = float(pc.x), float(pc.y)
    dx, dy = float(pd.x), float(pd.y)
    ex, ey = float(pe.x), float(pe.y)
    fx, fy = float(pf.x), float(pf.y)

    def dist(x1, y1, x2, y2) -> float:
        return math.hypot(x1 - x2, y1 - y2)

    denom = 2.0 * dist(ax, ay, dx, dy)
    if denom <= 1e-9:
        return None
    return (dist(bx, by, fx, fy) + dist(cx, cy, ex, ey)) / denom


def _eyes_closed_ear_avg(face_landmarks) -> float | None:
    ear_l = _ear(face_landmarks, 33, 160, 158, 133, 153, 144)
    ear_r = _ear(face_landmarks, 362, 385, 387, 263, 373, 380)
    ears = [v for v in (ear_l, ear_r) if v is not None]
    if not ears:
        return None
    return float(sum(ears) / len(ears))


def _level_label(level: int) -> str:
    return {
        0: "Focus 0",
        1: "Focus 1",
        2: "Focus 2",
        3: "Focus 3",
    }.get(level, f"unknown({level})")


def _level_color(level: int) -> tuple[int, int, int]:
    return {
        0: (0, 0, 255),
        1: (0, 165, 255),
        2: (0, 255, 255),
        3: (0, 255, 0),
    }.get(level, (255, 255, 255))


def _draw_hud(out, *, level: int, faces: int, fps: float, label: str, color: tuple[int, int, int]) -> None:
    h, w = out.shape[:2]
    x0, y0 = 10, 10
    x1, y1 = min(w - 10, 560), 110
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    out[:] = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
    cv2.putText(out, f"LEVEL {level}", (x0 + 10, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    cv2.putText(out, label, (x0 + 10, y0 + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        out,
        f"faces={faces} fps={fps:.1f}",
        (x0 + 10, y0 + 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )


def _camera_loop(
    model_path: Path,
    cam_id: int,
    seq_len: int,
    infer_interval: float,
    profile_every: int,
    infer_input_size: int,
    detect_max_width: int,
    capture_max_width: int,
    stream_max_width: int,
    jpeg_quality: int,
    det_interval: float,
) -> None:
    global _latest_jpeg

    load_error: str | None = None
    landmarker = None

    if not model_path.is_file():
        load_error = f"Model file not found: {model_path}"
        with _state_lock:
            _state["engine_ok"] = False
            _state["load_error"] = load_error
    else:
        try:
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
            from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=VisionTaskRunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
            )
            landmarker = FaceLandmarker.create_from_options(options)
            with _state_lock:
                _state["engine_ok"] = True
                _state["load_error"] = None
        except Exception:
            load_error = traceback.format_exc()
            with _state_lock:
                _state["engine_ok"] = False
                _state["load_error"] = load_error

    cap = cv2.VideoCapture(cam_id)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass
    try:
        req_w = max(320, int(capture_max_width))
        req_h = int(req_w * 9 / 16)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max(240, req_h))
    except Exception:
        pass
    if not cap.isOpened():
        err = f"Cannot open camera index={cam_id}"
        frame = _draw_error_frame(err + "\n" + (load_error or ""))
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            with _jpeg_lock:
                _latest_jpeg = buf.tobytes()
        return

    last_t = time.time()
    fps = 0.0
    last_ts_ms: int | None = None
    last_level: int | None = None
    smooth_yaw = 0.0
    smooth_pitch = 0.0
    smooth_roll = 0.0
    alpha = 0.35
    proxy_alpha = 0.25
    smooth_proxy: float | None = None
    proxy_baseline: float | None = None
    noface_count = 0
    pose_decay = 0.90
    eyes_closed_state = False
    eyes_closed_start_ms: int | None = None

    yaw_side_threshold_deg = 25.0
    pitch_down_threshold_deg = 20.0
    yaw_side_exit_threshold_deg = 18.0
    pitch_down_exit_threshold_deg = 12.0
    pitch_up_threshold_deg = 15.0
    pitch_up_exit_threshold_deg = 8.0
    down_proxy_delta_enter = 0.06
    down_proxy_delta_exit = 0.03
    noface_frames = 3
    eyes_closed_ear_threshold = 0.18
    eyes_closed_ear_exit_threshold = 0.21
    eyes_closed_seconds = 1.5
    roll_tilt_threshold_deg = 22.0
    roll_tilt_exit_threshold_deg = 14.0

    while not _stop.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = _resize_capture(frame, capture_max_width)
        frame = cv2.flip(frame, 1)

        if landmarker is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(time.time() * 1000)
            if last_ts_ms is not None and timestamp_ms <= last_ts_ms:
                timestamp_ms = last_ts_ms + 1
            last_ts_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms=timestamp_ms)
            out = frame.copy()
            faces = result.face_landmarks or []
            mats = getattr(result, "facial_transformation_matrixes", None) or []
            mat0 = mats[0] if len(mats) > 0 else None
            pose_ok = mat0 is not None
            level = 0 if len(faces) == 0 else 3

            if pose_ok:
                yaw, pitch, roll = _euler_yaw_pitch_roll_from_4x4(mat0)
                smooth_yaw = (1 - alpha) * smooth_yaw + alpha * yaw
                smooth_pitch = (1 - alpha) * smooth_pitch + alpha * pitch
                smooth_roll = (1 - alpha) * smooth_roll + alpha * roll
            else:
                smooth_yaw *= pose_decay
                smooth_pitch *= pose_decay
                smooth_roll *= pose_decay

            if len(faces) > 0:
                proxy = _down_proxy(faces[0])
                if proxy is not None:
                    smooth_proxy = proxy if smooth_proxy is None else (1 - proxy_alpha) * smooth_proxy + proxy_alpha * proxy
                    if (not pose_ok) or (abs(smooth_yaw) < yaw_side_exit_threshold_deg):
                        proxy_baseline = (
                            smooth_proxy if proxy_baseline is None else 0.98 * proxy_baseline + 0.02 * smooth_proxy
                        )

                ear_avg = _eyes_closed_ear_avg(faces[0])
                if ear_avg is not None:
                    if eyes_closed_state:
                        eyes_closed_state = ear_avg <= eyes_closed_ear_exit_threshold
                    else:
                        eyes_closed_state = ear_avg <= eyes_closed_ear_threshold

                if eyes_closed_state:
                    if eyes_closed_start_ms is None:
                        eyes_closed_start_ms = timestamp_ms
                else:
                    eyes_closed_start_ms = None
            else:
                eyes_closed_start_ms = None

            if len(faces) == 0:
                noface_count += 1
                if noface_count >= max(1, noface_frames):
                    level = 0
            else:
                noface_count = 0

            if len(faces) > 0 and eyes_closed_start_ms is not None:
                if (timestamp_ms - eyes_closed_start_ms) >= int(max(0.0, eyes_closed_seconds) * 1000):
                    level = 0

            if level != 0:
                side_by_yaw = False
                if pose_ok:
                    side_by_yaw = (last_level == 1 and abs(smooth_yaw) >= yaw_side_exit_threshold_deg) or (
                        abs(smooth_yaw) >= yaw_side_threshold_deg
                    )
                if side_by_yaw:
                    level = 1
                else:
                    down_by_pitch = False
                    up_by_pitch = False
                    tilt_by_roll = False
                    if pose_ok:
                        if last_level == 2:
                            down_by_pitch = smooth_pitch >= pitch_down_exit_threshold_deg
                            up_by_pitch = smooth_pitch <= -pitch_up_exit_threshold_deg
                            tilt_by_roll = abs(smooth_roll) >= roll_tilt_exit_threshold_deg
                        else:
                            down_by_pitch = smooth_pitch >= pitch_down_threshold_deg
                            up_by_pitch = smooth_pitch <= -pitch_up_threshold_deg
                            tilt_by_roll = abs(smooth_roll) >= roll_tilt_threshold_deg

                    down_by_proxy = False
                    if smooth_proxy is not None and proxy_baseline is not None:
                        delta = smooth_proxy - proxy_baseline
                        if last_level == 2:
                            down_by_proxy = delta >= down_proxy_delta_exit
                        else:
                            down_by_proxy = delta >= down_proxy_delta_enter

                    if down_by_pitch or down_by_proxy or up_by_pitch or tilt_by_roll:
                        level = 2
                    else:
                        level = 3

            last_level = level

            for face_landmarks in faces:
                for lm in face_landmarks:
                    x = int(lm.x * out.shape[1])
                    y = int(lm.y * out.shape[0])
                    cv2.circle(out, (x, y), 1, (0, 255, 0), -1, lineType=cv2.LINE_AA)

            now = time.time()
            dt = now - last_t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            last_t = now

            _draw_hud(
                out,
                level=level,
                faces=len(faces),
                fps=fps,
                label=_level_label(level),
                color=_level_color(level),
            )
            probs = [0.0, 0.0, 0.0, 0.0]
            probs[level] = 1.0
            _update_state_from_result(
                {
                    "face_detected": len(faces) > 0,
                    "cls_idx": int(level),
                    "conf": 1.0,
                    "probs": probs,
                }
            )
        else:
            out = frame
            h, w = out.shape[:2]
            overlay = _draw_error_frame(
                "Focus model not loaded\n" + (load_error or "Unknown error"),
                w=w,
                h=h,
            )
            out = cv2.addWeighted(out, 0.25, overlay, 0.75, 0)

        out = _resize_for_stream(out, stream_max_width)
        ok, buf = cv2.imencode(
            ".jpg",
            out,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                int(jpeg_quality),
                int(cv2.IMWRITE_JPEG_OPTIMIZE),
                1,
            ],
        )
        if ok:
            with _jpeg_lock:
                _latest_jpeg = buf.tobytes()

    cap.release()
    if landmarker is not None:
        landmarker.close()


def _mjpeg_gen():
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while not _stop.is_set():
        with _jpeg_lock:
            data = _latest_jpeg
        if data:
            yield boundary + data + b"\r\n"
        else:
            time.sleep(0.02)
    yield boundary


@app.route("/")
def index():
    with _state_lock:
        engine_ok = _state["engine_ok"]
        err = _state["load_error"]
    return render_template(
        "index.html",
        engine_ok=engine_ok,
        load_error_preview=(err[:500] + "…") if err and len(err) > 500 else err,
    )


@app.route("/api/state")
def api_state():
    with _state_lock:
        payload = dict(_state)
    
    # 检查是否需要更新AI策略
    ai_updated = _ai_manager.update_strategy_if_needed()
    
    # 获取AI策略和回复
    current_strategy = _ai_manager.get_current_strategy()
    ai_response = _ai_manager.get_last_ai_response()
    ai_report = _ai_manager.get_last_ai_report()
    
    # 计算精确的专注度值
    probs = payload.get("probs", [0.0, 0.0, 0.0, 0.0])
    cls_idx = payload.get("cls_idx")
    face_detected = payload.get("face_detected", False)
    precise_focus = _focus_soft_0_3(list(probs or []), face_detected)
    
    # 添加AI相关信息
    payload["ai_strategy"] = current_strategy
    payload["ai_response"] = ai_response
    payload["ai_report"] = ai_report
    payload["ai_updated"] = ai_updated
    payload["precise_focus"] = precise_focus
    
    # 兼容性字段：障碍高度等级 obstacle_h 与专注度同为 0–3（非 1–5）
    if cls_idx is None:
        strat, spd = "待机（未检测到人脸）", 1
    else:
        strategies = ("缓慢刺激", "轻度节奏", "平稳节奏", "高强度")
        strat = strategies[cls_idx] if 0 <= cls_idx < 4 else "未知"
        spd = max(1, min(5, cls_idx + 2))
    hgt = _obstacle_level_0_3(cls_idx, face_detected)
    payload["strategy"] = strat
    payload["speed"] = spd
    payload["obstacle_h"] = hgt
    
    return jsonify(payload)


@app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg_gen(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/game_summary", methods=["POST"])
def api_game_summary():
    payload = request.get_json(silent=True) or {}
    end_reason = str(payload.get("end_reason") or "游戏结束")
    duration_sec = float(payload.get("duration_sec") or 0.0)
    target_duration_sec = float(payload.get("target_duration_sec") or 0.0)
    mercy_count = int(payload.get("mercy_count") or 0)
    focus_values = payload.get("focus_values") or []
    if not isinstance(focus_values, list):
        focus_values = []
    summary = _ai_manager.generate_game_summary(
        end_reason=end_reason,
        duration_sec=max(0.0, duration_sec),
        target_duration_sec=max(0.0, target_duration_sec),
        mercy_count=max(0, mercy_count),
        focus_values=[float(v) for v in focus_values[:4000] if isinstance(v, (int, float))],
    )
    return jsonify({"ok": True, "summary": summary})


def _start_camera_thread(
    model_path: Path,
    cam_id: int,
    seq_len: int,
    infer_interval: float,
    profile_every: int,
    infer_input_size: int,
    detect_max_width: int,
    capture_max_width: int,
    stream_max_width: int,
    jpeg_quality: int,
    det_interval: float,
) -> None:
    global _cam_thread
    if _cam_thread is not None and _cam_thread.is_alive():
        return
    _stop.clear()
    _cam_thread = threading.Thread(
        target=_camera_loop,
        args=(
            model_path,
            cam_id,
            seq_len,
            infer_interval,
            profile_every,
            infer_input_size,
            detect_max_width,
            capture_max_width,
            stream_max_width,
            jpeg_quality,
            det_interval,
        ),
        daemon=True,
    )
    _cam_thread.start()


def create_app(
    model_path: Path | None = None,
    cam_id: int = 0,
    seq_len: int = 12,
    infer_interval: float = 0.25,
    profile_every: int = 5,
    infer_input_size: int = 160,
    detect_max_width: int = 360,
    capture_max_width: int = 960,
    stream_max_width: int = 720,
    jpeg_quality: int = 75,
    det_interval: float = 0.065,
) -> Flask:
    model_file = (model_path or default_model_path()).resolve()
    _start_camera_thread(
        model_file,
        cam_id=cam_id,
        seq_len=max(1, seq_len),
        infer_interval=max(0.05, float(infer_interval)),
        profile_every=max(1, int(profile_every)),
        infer_input_size=max(96, min(512, int(infer_input_size))),
        detect_max_width=max(0, int(detect_max_width)),
        capture_max_width=max(0, int(capture_max_width)),
        stream_max_width=max(0, int(stream_max_width)),
        jpeg_quality=max(40, min(95, int(jpeg_quality))),
        det_interval=max(0.02, float(det_interval)),
    )
    return app


def parse_args():
    p = argparse.ArgumentParser(description="专注度训练游戏 — Flask 网页")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--cam-id", type=int, default=0)
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="规则模型 .task 路径，默认 rule_model/models/face_landmarker.task",
    )
    p.add_argument(
        "--seq-len",
        type=int,
        default=12,
        help="时序长度 T，略小可提速，默认 12",
    )
    p.add_argument(
        "--infer-interval",
        type=float,
        default=0.25,
        help="两次神经网络推理的最小间隔（秒），越大越省算力，默认 0.25",
    )
    p.add_argument(
        "--profile-every",
        type=int,
        default=5,
        help="每隔多少帧才跑一次侧脸 Haar，默认 5",
    )
    p.add_argument(
        "--infer-input-size",
        type=int,
        default=160,
        help="送入 ResNet 的人脸边长（像素），224 最准但更慢，默认 160",
    )
    p.add_argument(
        "--detect-width",
        type=int,
        default=360,
        help="人脸检测用图最大宽度，0 表示与采集同尺寸，默认 360",
    )
    p.add_argument(
        "--capture-width",
        type=int,
        default=960,
        help="摄像头采集后先缩到此宽度再进行规则推理，0 不缩，默认 960",
    )
    p.add_argument(
        "--stream-width",
        type=int,
        default=720,
        help="推流 MJPEG 最大宽度，默认 720（更清晰）",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=75,
        help="MJPEG JPEG 质量 40–95，默认 75（更清晰）",
    )
    p.add_argument(
        "--det-interval",
        type=float,
        default=0.065,
        help="每隔多少秒做一次完整人脸检测（Haar）；中间帧用上一框快速跟踪并推流，默认 0.065",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_file = args.model
    if model_file is None:
        model_file = default_model_path()
    model_file = model_file.resolve()
    print(f"项目根: {project_root()}")
    print(f"模型路径: {model_file}  exists={model_file.is_file()}")
    print(
        "性能参数: "
        f"capture_w={args.capture_width} detect_w={args.detect_width} "
        f"infer_px={args.infer_input_size} seq={args.seq_len} "
        f"infer_iv={args.infer_interval}s det_iv={args.det_interval}s "
        f"stream={args.stream_width} jpeg={args.jpeg_quality}"
    )

    create_app(
        model_path=model_file,
        cam_id=args.cam_id,
        seq_len=max(1, args.seq_len),
        infer_interval=args.infer_interval,
        profile_every=args.profile_every,
        infer_input_size=args.infer_input_size,
        detect_max_width=args.detect_width,
        capture_max_width=args.capture_width,
        stream_max_width=args.stream_width,
        jpeg_quality=args.jpeg_quality,
        det_interval=args.det_interval,
    )
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
