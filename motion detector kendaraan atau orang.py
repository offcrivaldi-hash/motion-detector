from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - handled at runtime for friendlier UX
    YOLO = None

try:
    import torch
except ImportError:  # pragma: no cover - handled at runtime for friendlier UX
    torch = None


DEFAULT_ROI_NORM = (0.115, 0.163, 0.837, 0.906)
MOTORCYCLE_CLASS_ID = 3
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    hits: int
    missed: int
    best_conf: float
    first_center: Tuple[float, float]
    last_center: Tuple[float, float]
    first_frame: int
    last_frame: int
    event_saved: bool = False


@dataclass
class SavedEvent:
    frame_ms: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]


@dataclass
class VideoJob:
    video_path: Path
    base_datetime: datetime | None
    time_source: str


@dataclass
class VideoProcessResult:
    video_path: Path
    output_dir: Path
    log_path: Path
    base_datetime: datetime | None
    time_source: str
    total_frames: int
    processed_frames: int
    last_frame_index: int
    events_saved: int
    detection_rows: List[List[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deteksi motor untuk video lokal dengan background subtraction + YOLO di area gerak."
    )
    parser.add_argument("--video", help="Path video lokal yang akan diproses.")
    parser.add_argument(
        "--input-dir",
        help="Folder video yang akan diproses semua. Kalau diisi, program scan seluruh video di folder ini.",
    )
    parser.add_argument(
        "--output-dir",
        default="motor_output",
        help="Folder output root untuk screenshot dan log.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Path file model YOLO atau nama model Ultralytics.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Perangkat inferensi YOLO, contoh: auto, cpu, 0, 0,1.",
    )
    parser.add_argument(
        "--start-datetime",
        default=None,
        help="Waktu awal video tunggal, format 'YYYY-MM-DD HH:MM:SS'. Dipakai untuk timestamp file/log.",
    )
    parser.add_argument(
        "--video-time-source",
        choices=("auto", "map", "filename", "mtime", "none"),
        default="auto",
        help="Sumber waktu awal video untuk mode folder. 'auto' = map > filename > modified time.",
    )
    parser.add_argument(
        "--video-time-map",
        default=None,
        help="CSV mapping waktu awal video. Kolom minimal: file,start_datetime",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subfolder juga saat memakai --input-dir.",
    )
    parser.add_argument(
        "--roi-norm",
        default=",".join(str(v) for v in DEFAULT_ROI_NORM),
        help="ROI normalisasi x1,y1,x2,y2 sesuai area merah. Contoh: 0.115,0.163,0.837,0.906",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=1,
        help="Mulai proses dari frame tertentu. Berguna untuk testing segmen video.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Jumlah frame awal setelah seek yang dipakai untuk adaptasi background tanpa YOLO/event.",
    )
    parser.add_argument(
        "--motion-history",
        type=int,
        default=500,
        help="History background subtractor MOG2.",
    )
    parser.add_argument(
        "--motion-var-threshold",
        type=float,
        default=32.0,
        help="Var threshold background subtractor.",
    )
    parser.add_argument(
        "--min-motion-area",
        type=int,
        default=2500,
        help="Luas contour minimum untuk dianggap gerakan.",
    )
    parser.add_argument(
        "--motion-padding",
        type=int,
        default=24,
        help="Padding piksel di sekitar contour gerakan sebelum dikirim ke YOLO.",
    )
    parser.add_argument(
        "--motion-scale",
        type=float,
        default=1.0,
        help="Skala resize ROI sebelum background subtraction. Contoh 0.5 = 2x lebih kecil.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Proses 1 frame tiap N frame. Contoh 5 = hanya proses frame ke-1,6,11,...",
    )
    parser.add_argument(
        "--max-motion-boxes",
        type=int,
        default=0,
        help="Batasi jumlah kandidat area gerak yang dikirim ke YOLO. 0 = tanpa batas.",
    )
    parser.add_argument(
        "--yolo-conf",
        type=float,
        default=0.35,
        help="Confidence minimum deteksi YOLO.",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=640,
        help="Ukuran inferensi YOLO.",
    )
    parser.add_argument(
        "--track-iou",
        type=float,
        default=0.25,
        help="Threshold IoU untuk menyambung deteksi ke track lama.",
    )
    parser.add_argument(
        "--track-max-missed",
        type=int,
        default=12,
        help="Jumlah frame hilang sebelum track dibuang.",
    )
    parser.add_argument(
        "--track-min-hits",
        type=int,
        default=2,
        help="Jumlah konfirmasi minimum sebelum event screenshot dibuat.",
    )
    parser.add_argument(
        "--track-min-travel",
        type=float,
        default=40.0,
        help="Jarak perpindahan minimum (piksel) sebelum event dianggap motor lewat.",
    )
    parser.add_argument(
        "--event-dedupe-ms",
        type=float,
        default=900.0,
        help="Jeda maksimum antar event yang masih dianggap motor yang sama.",
    )
    parser.add_argument(
        "--event-dedupe-iou",
        type=float,
        default=0.15,
        help="IoU minimum untuk menganggap dua event adalah motor yang sama.",
    )
    parser.add_argument(
        "--event-dedupe-distance",
        type=float,
        default=180.0,
        help="Jarak pusat maksimum untuk menganggap dua event adalah motor yang sama.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Batasi jumlah frame untuk testing cepat.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Tampilkan preview realtime selama proses.",
    )
    parser.add_argument(
        "--progress-step",
        type=float,
        default=5.0,
        help="Interval persen untuk menampilkan progress. Default 5 persen.",
    )
    return parser.parse_args()


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError("Format --start-datetime harus 'YYYY-MM-DD HH:MM:SS'.")


def parse_roi_norm(value: str) -> Tuple[float, float, float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi-norm harus berisi 4 angka: x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("Nilai --roi-norm harus di rentang 0..1 dan x1<x2, y1<y2.")
    return x1, y1, x2, y2


def norm_roi_to_pixels(
    roi_norm: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi_norm
    return (
        int(x1 * width),
        int(y1 * height),
        int(x2 * width),
        int(y2 * height),
    )


def clamp_box(
    bbox: Tuple[int, int, int, int], width: int, height: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(1, min(width, x2)),
        max(1, min(height, y2)),
    )


def box_area(bbox: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = box_area((inter_x1, inter_y1, inter_x2, inter_y2))
    if inter_area <= 0:
        return 0.0
    union_area = box_area(box_a) + box_area(box_b) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def box_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def merge_boxes(boxes: Iterable[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    merged: List[Tuple[int, int, int, int]] = []
    for box in boxes:
        current = box
        changed = True
        while changed:
            changed = False
            next_merged: List[Tuple[int, int, int, int]] = []
            for existing in merged:
                if compute_iou(current, existing) > 0 or boxes_touch(current, existing):
                    current = (
                        min(current[0], existing[0]),
                        min(current[1], existing[1]),
                        max(current[2], existing[2]),
                        max(current[3], existing[3]),
                    )
                    changed = True
                else:
                    next_merged.append(existing)
            merged = next_merged
        merged.append(current)
    return merged


def boxes_touch(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int], gap: int = 18) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 + gap < bx1 or bx2 + gap < ax1 or ay2 + gap < by1 or by2 + gap < ay1)


def motion_boxes(
    frame: np.ndarray,
    roi_px: Tuple[int, int, int, int],
    subtractor: cv2.BackgroundSubtractor,
    min_motion_area: int,
    padding: int,
    motion_scale: float,
    max_motion_boxes: int,
) -> List[Tuple[int, int, int, int]]:
    frame_h, frame_w = frame.shape[:2]
    rx1, ry1, rx2, ry2 = roi_px
    roi = frame[ry1:ry2, rx1:rx2]
    scale = motion_scale if 0 < motion_scale <= 1.0 else 1.0
    scaled_roi = roi
    if scale < 1.0:
        scaled_roi = cv2.resize(roi, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    fgmask = subtractor.apply(scaled_roi)
    _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(cleaned, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        scaled_min_area = max(50.0, float(min_motion_area) * (scale ** 2))
        if area < scaled_min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if scale < 1.0:
            x = int(round(x / scale))
            y = int(round(y / scale))
            w = int(round(w / scale))
            h = int(round(h / scale))
        box = clamp_box(
            (rx1 + x - padding, ry1 + y - padding, rx1 + x + w + padding, ry1 + y + h + padding),
            frame_w,
            frame_h,
        )
        if box_area(box) > 0:
            candidates.append(box)
    merged = merge_boxes(candidates)
    if max_motion_boxes and max_motion_boxes > 0:
        merged = sorted(merged, key=box_area, reverse=True)[:max_motion_boxes]
    return merged


def dedupe_boxes(
    detections: Sequence[Tuple[int, int, int, int, float]],
    iou_threshold: float = 0.45,
) -> List[Tuple[int, int, int, int, float]]:
    ordered = sorted(detections, key=lambda item: item[4], reverse=True)
    kept: List[Tuple[int, int, int, int, float]] = []
    for candidate in ordered:
        cand_box = candidate[:4]
        if any(compute_iou(cand_box, kept_item[:4]) >= iou_threshold for kept_item in kept):
            continue
        kept.append(candidate)
    return kept


def run_yolo_on_motion_regions(
    model: YOLO,
    frame: np.ndarray,
    candidate_boxes: Sequence[Tuple[int, int, int, int]],
    conf_threshold: float,
    imgsz: int,
    device: str,
) -> List[Tuple[int, int, int, int, float]]:
    frame_h, frame_w = frame.shape[:2]
    detections: List[Tuple[int, int, int, int, float]] = []
    for mx1, my1, mx2, my2 in candidate_boxes:
        crop = frame[my1:my2, mx1:mx2]
        if crop.size == 0:
            continue
        results = model.predict(
            source=crop,
            classes=[MOTORCYCLE_CLASS_ID],
            conf=conf_threshold,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        if not results:
            continue
        result = results[0]
        if result.boxes is None:
            continue
        for xyxy, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = xyxy.tolist()
            full_box = clamp_box(
                (int(mx1 + x1), int(my1 + y1), int(mx1 + x2), int(my1 + y2)),
                frame_w,
                frame_h,
            )
            detections.append((*full_box, float(conf)))
    return dedupe_boxes(detections)


def update_tracks(
    tracks: dict[int, Track],
    detections: Sequence[Tuple[int, int, int, int, float]],
    frame_index: int,
    iou_threshold: float,
    max_missed: int,
    next_track_id: int,
) -> int:
    unmatched_tracks = set(tracks.keys())
    unmatched_detections = set(range(len(detections)))
    matches: List[Tuple[float, int, int]] = []

    for track_id, track in tracks.items():
        for det_index, detection in enumerate(detections):
            score = compute_iou(track.bbox, detection[:4])
            if score >= iou_threshold:
                matches.append((score, track_id, det_index))

    matches.sort(reverse=True)
    used_tracks: set[int] = set()
    used_detections: set[int] = set()

    for _, track_id, det_index in matches:
        if track_id in used_tracks or det_index in used_detections:
            continue
        track = tracks[track_id]
        det_bbox = detections[det_index][:4]
        det_conf = detections[det_index][4]
        track.bbox = det_bbox
        track.hits += 1
        track.missed = 0
        track.best_conf = max(track.best_conf, det_conf)
        track.last_center = box_center(det_bbox)
        track.last_frame = frame_index
        unmatched_tracks.discard(track_id)
        unmatched_detections.discard(det_index)
        used_tracks.add(track_id)
        used_detections.add(det_index)

    for track_id in list(unmatched_tracks):
        track = tracks[track_id]
        track.missed += 1
        if track.missed > max_missed:
            tracks.pop(track_id, None)

    for det_index in unmatched_detections:
        det_bbox = detections[det_index][:4]
        det_conf = detections[det_index][4]
        center = box_center(det_bbox)
        tracks[next_track_id] = Track(
            track_id=next_track_id,
            bbox=det_bbox,
            hits=1,
            missed=0,
            best_conf=det_conf,
            first_center=center,
            last_center=center,
            first_frame=frame_index,
            last_frame=frame_index,
        )
        next_track_id += 1

    return next_track_id


def timestamp_strings(base_datetime: datetime | None, frame_ms: float) -> Tuple[str, str]:
    if base_datetime is not None:
        dt = base_datetime + timedelta(milliseconds=frame_ms)
        display = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        stamp = dt.strftime("%Y%m%d_%H%M%S") + f"_{dt.microsecond // 1000:03d}"
        return display, stamp

    total_ms = int(round(frame_ms))
    total_seconds, milliseconds = divmod(total_ms, 1000)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    display = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    stamp = f"nodate_{hours:02d}{minutes:02d}{seconds:02d}_{milliseconds:03d}"
    return display, stamp





def save_event(
    frame: np.ndarray,
    roi_px: Tuple[int, int, int, int],
    track: Track,
    frame_index: int,
    frame_ms: float,
    base_datetime: datetime | None,
    output_dir: Path,
) -> Tuple[Path, str]:
    display_ts, stamp_ts = timestamp_strings(base_datetime, frame_ms)
    x1, y1, x2, y2 = clamp_box(track.bbox, frame.shape[1], frame.shape[0])

    full_name = f"motor_{stamp_ts}.jpg"
    full_path = output_dir / full_name

    annotated = frame.copy()
    cv2.rectangle(annotated, (roi_px[0], roi_px[1]), (roi_px[2], roi_px[3]), (0, 0, 255), 2)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
    label = f"motor id={track.track_id} conf={track.best_conf:.2f} {display_ts}"
    label_y = max(30, y1 - 12)
    cv2.putText(
        annotated,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(full_path), annotated)
    return full_path, display_ts


def is_duplicate_event(
    saved_events: Sequence[SavedEvent],
    bbox: Tuple[int, int, int, int],
    frame_ms: float,
    dedupe_ms: float,
    dedupe_iou: float,
    dedupe_distance: float,
) -> bool:
    center = box_center(bbox)
    for event in reversed(saved_events):
        if frame_ms - event.frame_ms > dedupe_ms:
            break
        if compute_iou(bbox, event.bbox) >= dedupe_iou:
            return True
        if distance(center, event.center) <= dedupe_distance:
            return True
    return False


def ensure_model_ready(model_name: str) -> None:
    if YOLO is None:
        raise RuntimeError(
            "Paket 'ultralytics' belum terpasang. Install dulu dengan: pip install ultralytics"
        )
    if model_name.endswith(".pt") and not Path(model_name).exists():
        # Ultralytics masih boleh mengunduh model bawaan seperti yolov8n.pt.
        # Validasi file lokal hanya dilakukan bila nama model berisi path folder.
        parent = Path(model_name).parent
        if str(parent) not in ("", ".") and not parent.exists():
            raise FileNotFoundError(f"Folder model tidak ditemukan: {parent}")


def resolve_inference_device(requested_device: str) -> Tuple[str, str]:
    if requested_device != "auto":
        return requested_device, f"manual:{requested_device}"

    if torch is None:
        return "cpu", "auto:torch_not_installed"

    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            gpu_name = torch.cuda.get_device_name(0)
            return "0", f"auto:cuda:{gpu_name}"
    except Exception:
        return "cpu", "auto:cuda_check_failed"

    cuda_build = getattr(getattr(torch, "version", None), "cuda", None)
    if cuda_build:
        return "cpu", f"auto:cuda_runtime_unavailable(build_cuda={cuda_build})"
    return "cpu", "auto:cpu_only_torch"


def runtime_backend_summary(resolved_device: str, device_reason: str) -> str:
    cpu_summary = "CPU untuk decode video, background subtraction, tracking, simpan file"
    if resolved_device == "cpu":
        return f"{cpu_summary}; YOLO=CPU ({device_reason})"
    return f"{cpu_summary}; YOLO=GPU device {resolved_device} ({device_reason})"


def draw_preview(
    frame: np.ndarray,
    roi_px: Tuple[int, int, int, int],
    motion_candidates: Sequence[Tuple[int, int, int, int]],
    detections: Sequence[Tuple[int, int, int, int, float]],
    tracks: dict[int, Track],
) -> np.ndarray:
    preview = frame.copy()
    cv2.rectangle(preview, (roi_px[0], roi_px[1]), (roi_px[2], roi_px[3]), (0, 0, 255), 2)
    for x1, y1, x2, y2 in motion_candidates:
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 180, 255), 2)
    for x1, y1, x2, y2, conf in detections:
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            preview,
            f"{conf:.2f}",
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    for track in tracks.values():
        cx, cy = track.last_center
        cv2.putText(
            preview,
            f"T{track.track_id} h{track.hits}",
            (int(cx) + 6, int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return preview


def parse_datetime_from_filename(name: str) -> datetime | None:
    patterns = [
        r"(?P<y>\d{4})[-_]?((?P<m>\d{2}))[-_]?((?P<d>\d{2}))[T _-]?(?P<h>\d{2})[:_-]?(?P<mi>\d{2})[:_-]?(?P<s>\d{2})",
        r"(?P<d>\d{2})[-_](?P<m>\d{2})[-_](?P<y>\d{4})[ T_-]?(?P<h>\d{2})[:_-]?(?P<mi>\d{2})[:_-]?(?P<s>\d{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if not match:
            continue
        parts = {key: int(value) for key, value in match.groupdict().items()}
        try:
            return datetime(parts["y"], parts["m"], parts["d"], parts["h"], parts["mi"], parts["s"])
        except ValueError:
            continue
    return None


def load_video_time_map(csv_path: str | None) -> dict[str, datetime]:
    if not csv_path:
        return {}
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"File --video-time-map tidak ditemukan: {path}")

    mapping: dict[str, datetime] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV --video-time-map kosong atau header tidak valid.")

        file_key = next(
            (field for field in reader.fieldnames if field.lower() in {"file", "filename", "video", "path"}),
            None,
        )
        time_key = next(
            (field for field in reader.fieldnames if field.lower() in {"start_datetime", "datetime", "video_datetime"}),
            None,
        )
        if not file_key or not time_key:
            raise ValueError("CSV --video-time-map butuh kolom file dan start_datetime.")

        for row in reader:
            raw_file = (row.get(file_key) or "").strip()
            raw_dt = (row.get(time_key) or "").strip()
            if not raw_file or not raw_dt:
                continue
            dt = parse_datetime(raw_dt)
            if dt is None:
                raise ValueError(f"Datetime tidak valid di --video-time-map untuk file '{raw_file}'.")
            file_path = Path(raw_file)
            for key in {raw_file.lower(), file_path.name.lower(), file_path.stem.lower()}:
                mapping[key] = dt
    return mapping


def resolve_video_time_from_map(video_path: Path, mapping: dict[str, datetime]) -> datetime | None:
    candidates = [
        str(video_path).lower(),
        str(video_path.resolve()).lower(),
        video_path.name.lower(),
        video_path.stem.lower(),
    ]
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return None


def resolve_video_base_datetime(
    video_path: Path,
    args: argparse.Namespace,
    explicit_start: datetime | None,
    mapping: dict[str, datetime],
    folder_mode: bool,
) -> Tuple[datetime | None, str]:
    if not folder_mode and explicit_start is not None:
        return explicit_start, "arg"

    source = args.video_time_source
    if source in {"auto", "map"}:
        mapped = resolve_video_time_from_map(video_path, mapping)
        if mapped is not None:
            return mapped, "map"
        if source == "map":
            return None, "none"

    if source in {"auto", "filename"}:
        parsed = parse_datetime_from_filename(video_path.stem)
        if parsed is not None:
            return parsed, "filename"
        if source == "filename":
            return None, "none"

    if source in {"auto", "mtime"}:
        return datetime.fromtimestamp(video_path.stat().st_mtime), "mtime"

    return None, "none"


def discover_video_jobs(
    args: argparse.Namespace,
    explicit_start: datetime | None,
    mapping: dict[str, datetime],
) -> List[VideoJob]:
    jobs: List[VideoJob] = []
    if args.video:
        video_path = Path(args.video)
        base_datetime, time_source = resolve_video_base_datetime(
            video_path=video_path,
            args=args,
            explicit_start=explicit_start,
            mapping=mapping,
            folder_mode=False,
        )
        return [VideoJob(video_path=video_path, base_datetime=base_datetime, time_source=time_source)]

    input_dir = Path(args.input_dir)
    iterator = input_dir.rglob("*") if args.recursive else input_dir.glob("*")
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        base_datetime, time_source = resolve_video_base_datetime(
            video_path=path,
            args=args,
            explicit_start=explicit_start,
            mapping=mapping,
            folder_mode=True,
        )
        jobs.append(VideoJob(video_path=path, base_datetime=base_datetime, time_source=time_source))

    jobs.sort(
        key=lambda job: (
            0 if job.base_datetime is not None else 1,
            job.base_datetime or datetime.max,
            job.video_path.name.lower(),
        )
    )
    return jobs


def make_video_output_dir(root_output_dir: Path, video_path: Path, folder_mode: bool) -> Path:
    if not folder_mode:
        return root_output_dir
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", video_path.stem).strip("._") or video_path.stem
    return root_output_dir / safe_name


def process_video(
    video_path: Path,
    output_dir: Path,
    base_datetime: datetime | None,
    time_source: str,
    roi_norm: Tuple[float, float, float, float],
    args: argparse.Namespace,
    model: YOLO,
    resolved_device: str,
    root_output_dir: Path,
    job_index: int,
    total_jobs: int,
) -> VideoProcessResult:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Gagal membuka video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    roi_px = norm_roi_to_pixels(roi_norm, width, height)

    if args.start_frame > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame - 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "detections.txt"

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=args.motion_history,
        varThreshold=args.motion_var_threshold,
        detectShadows=True,
    )

    tracks: dict[int, Track] = {}
    saved_events: List[SavedEvent] = []
    next_track_id = 1
    frame_index = args.start_frame - 1
    processed_frames = 0
    events_saved = 0
    detection_rows: List[List[str]] = []
    window_name = f"Motor Detector - {video_path.name}"
    progress_every_n_frames = progress_frame_interval(
        total_frames=total_frames or max(args.start_frame, 1),
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        progress_step=args.progress_step,
    )
    progress_total_window_frames = progress_window_total_frames(
        total_frames=total_frames or max(args.start_frame, 1),
        start_frame=args.start_frame,
        max_frames=args.max_frames,
    )
    last_progress_bucket = -1
    last_progress_frame = -1

    with log_path.open("w", encoding="utf-8") as log_file:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_frames and processed_frames >= args.max_frames:
                break

            frame_index += 1
            if (
                total_frames > 0
                and frame_index - last_progress_frame >= progress_every_n_frames
            ):
                progress = compute_progress_percent(
                    frame_index=frame_index,
                    start_frame=args.start_frame,
                    total_frames=total_frames,
                    max_frames=args.max_frames,
                )
                progress_bucket = int(progress / max(0.1, args.progress_step))
                if progress_bucket > last_progress_bucket:
                    print(
                        f"[PROGRESS] ({job_index}/{total_jobs}) video={video_path.name} "
                        f"{progress:5.1f}% frame={frame_index - args.start_frame + 1}/{progress_total_window_frames} "
                        f"events={events_saved}"
                    )
                    last_progress_bucket = progress_bucket
                    last_progress_frame = frame_index
            relative_frame_index = frame_index - args.start_frame
            should_process_frame = (relative_frame_index % args.frame_step) == 0
            if not should_process_frame:
                continue
            processed_frames += 1

            frame_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if frame_ms <= 0:
                frame_ms = (frame_index - 1) * (1000.0 / fps)

            warmup_limit = args.start_frame - 1 + args.warmup_frames
            if frame_index <= warmup_limit:
                warmup_roi = frame[roi_px[1]:roi_px[3], roi_px[0]:roi_px[2]]
                if args.motion_scale < 1.0:
                    warmup_roi = cv2.resize(
                        warmup_roi,
                        dsize=None,
                        fx=args.motion_scale,
                        fy=args.motion_scale,
                        interpolation=cv2.INTER_AREA,
                    )
                subtractor.apply(warmup_roi)
                if args.show:
                    preview = draw_preview(
                        frame=frame,
                        roi_px=roi_px,
                        motion_candidates=[],
                        detections=[],
                        tracks=tracks,
                    )
                    cv2.putText(
                        preview,
                        f"{video_path.name} warmup frame {frame_index}/{warmup_limit}",
                        (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window_name, preview)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                continue

            motion_candidates = motion_boxes(
                frame=frame,
                roi_px=roi_px,
                subtractor=subtractor,
                min_motion_area=args.min_motion_area,
                padding=args.motion_padding,
                motion_scale=args.motion_scale,
                max_motion_boxes=args.max_motion_boxes,
            )

            detections: List[Tuple[int, int, int, int, float]] = []
            if motion_candidates:
                detections = run_yolo_on_motion_regions(
                    model=model,
                    frame=frame,
                    candidate_boxes=motion_candidates,
                    conf_threshold=args.yolo_conf,
                    imgsz=args.yolo_imgsz,
                    device=resolved_device,
                )

            next_track_id = update_tracks(
                tracks=tracks,
                detections=detections,
                frame_index=frame_index,
                iou_threshold=args.track_iou,
                max_missed=args.track_max_missed,
                next_track_id=next_track_id,
            )

            for track in tracks.values():
                if track.event_saved:
                    continue
                if track.hits < args.track_min_hits:
                    continue
                if distance(track.first_center, track.last_center) < args.track_min_travel:
                    continue
                if is_duplicate_event(
                    saved_events=saved_events,
                    bbox=track.bbox,
                    frame_ms=frame_ms,
                    dedupe_ms=args.event_dedupe_ms,
                    dedupe_iou=args.event_dedupe_iou,
                    dedupe_distance=args.event_dedupe_distance,
                ):
                    track.event_saved = True
                    continue

                full_path, display_ts = save_event(
                    frame=frame,
                    roi_px=roi_px,
                    track=track,
                    frame_index=frame_index,
                    frame_ms=frame_ms,
                    base_datetime=base_datetime,
                    output_dir=output_dir,
                )
                log_line = (
                    f"{display_ts} | frame={frame_index} track={track.track_id} "
                    f"conf={track.best_conf:.4f} bbox={track.bbox} file={full_path.name}\n"
                )
                log_file.write(log_line)
                log_file.flush()
                track.event_saved = True
                saved_events.append(
                    SavedEvent(
                        frame_ms=frame_ms,
                        bbox=track.bbox,
                        center=box_center(track.bbox),
                    )
                )
                while saved_events and frame_ms - saved_events[0].frame_ms > args.event_dedupe_ms:
                    saved_events.pop(0)
                events_saved += 1
                print(
                    f"[EVENT] video={video_path.name} frame={frame_index} track={track.track_id} "
                    f"time={display_ts} file={full_path.name}"
                )

            if args.show:
                preview = draw_preview(
                    frame=frame,
                    roi_px=roi_px,
                    motion_candidates=motion_candidates,
                    detections=detections,
                    tracks=tracks,
                )
                cv2.putText(
                    preview,
                    video_path.name,
                    (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, preview)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    cap.release()
    if args.show:
        cv2.destroyWindow(window_name)

    print(
        f"[DONE] video={video_path.name} source={time_source} "
        f"start={base_datetime.strftime('%Y-%m-%d %H:%M:%S') if base_datetime else 'N/A'} "
        f"frame_terakhir={frame_index} diproses={processed_frames} event_motor={events_saved} log={log_path}"
    )
    return VideoProcessResult(
        video_path=video_path,
        output_dir=output_dir,
        log_path=log_path,
        base_datetime=base_datetime,
        time_source=time_source,
        total_frames=total_frames,
        processed_frames=processed_frames,
        last_frame_index=frame_index,
        events_saved=events_saved,
        detection_rows=[],
    )





def compute_progress_percent(
    frame_index: int,
    start_frame: int,
    total_frames: int,
    max_frames: int | None,
) -> float:
    if max_frames:
        window_end = min(total_frames, start_frame - 1 + max_frames)
    else:
        window_end = total_frames
    window_total = max(1, window_end - start_frame + 1)
    completed = min(window_total, max(0, frame_index - start_frame + 1))
    return min(100.0, max(0.0, (completed / window_total) * 100.0))


def progress_frame_interval(
    total_frames: int,
    start_frame: int,
    max_frames: int | None,
    progress_step: float,
) -> int:
    if max_frames:
        window_end = min(total_frames, start_frame - 1 + max_frames)
    else:
        window_end = total_frames
    window_total = max(1, window_end - start_frame + 1)
    step_ratio = max(0.1, progress_step) / 100.0
    return max(1, int(window_total * step_ratio))


def progress_window_total_frames(total_frames: int, start_frame: int, max_frames: int | None) -> int:
    if max_frames:
        window_end = min(total_frames, start_frame - 1 + max_frames)
    else:
        window_end = total_frames
    return max(1, window_end - start_frame + 1)


def main() -> int:
    args = parse_args()

    try:
        explicit_start = parse_datetime(args.start_datetime)
        roi_norm = parse_roi_norm(args.roi_norm)
        video_time_map = load_video_time_map(args.video_time_map)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if not args.video and not args.input_dir:
        print("[ERROR] Isi salah satu: --video atau --input-dir", file=sys.stderr)
        return 2
    if args.video and args.input_dir:
        print("[ERROR] Pilih salah satu saja: --video atau --input-dir", file=sys.stderr)
        return 2
    if args.start_frame < 1:
        print("[ERROR] --start-frame minimal 1.", file=sys.stderr)
        return 2
    if args.warmup_frames < 0:
        print("[ERROR] --warmup-frames tidak boleh negatif.", file=sys.stderr)
        return 2
    if args.frame_step < 1:
        print("[ERROR] --frame-step minimal 1.", file=sys.stderr)
        return 2
    if not (0 < args.motion_scale <= 1.0):
        print("[ERROR] --motion-scale harus di rentang >0 sampai 1.0.", file=sys.stderr)
        return 2
    if args.input_dir and explicit_start is not None:
        print(
            "[ERROR] --start-datetime khusus video tunggal. Untuk mode folder pakai --video-time-map atau --video-time-source.",
            file=sys.stderr,
        )
        return 2

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"[ERROR] Video tidak ditemukan: {video_path}", file=sys.stderr)
            return 2
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"[ERROR] Folder video tidak ditemukan: {input_dir}", file=sys.stderr)
            return 2

    jobs = discover_video_jobs(args=args, explicit_start=explicit_start, mapping=video_time_map)
    if not jobs:
        print("[ERROR] Tidak ada file video yang ditemukan.", file=sys.stderr)
        return 2

    ensure_model_ready(args.model)
    resolved_device, device_reason = resolve_inference_device(args.device)
    model = YOLO(args.model)

    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)
    folder_mode = args.input_dir is not None

    print(f"[RUNTIME] {runtime_backend_summary(resolved_device, device_reason)}")
    print(f"[INFO] Video ditemukan: {len(jobs)}")
    for index, job in enumerate(jobs, start=1):
        resolved = job.base_datetime.strftime("%Y-%m-%d %H:%M:%S") if job.base_datetime else "N/A"
        print(f"[QUEUE] {index:03d} {job.video_path.name} start={resolved} source={job.time_source}")

    results: List[VideoProcessResult] = []
    for index, job in enumerate(jobs, start=1):
        print(f"[START] ({index}/{len(jobs)}) {job.video_path}")
        output_dir = make_video_output_dir(root_output_dir, job.video_path, folder_mode=folder_mode)
        try:
            result = process_video(
                video_path=job.video_path,
                output_dir=output_dir,
                base_datetime=job.base_datetime,
                time_source=job.time_source,
                roi_norm=roi_norm,
                args=args,
                model=model,
                resolved_device=resolved_device,
                root_output_dir=root_output_dir,
                job_index=index,
                total_jobs=len(jobs),
            )
        except RuntimeError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2
        results.append(result)

    if folder_mode:
        total_events = sum(r.events_saved for r in results)
        print(
            f"[DONE] Batch selesai. video={len(results)} total_event_motor={total_events} output={root_output_dir}"
        )
    elif results:
        only = results[0]
        print(
            f"[DONE] Proses selesai. Frame terakhir={only.last_frame_index}, diproses={only.processed_frames}, "
            f"event motor={only.events_saved}, log={only.log_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
