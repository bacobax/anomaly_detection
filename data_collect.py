# Threaded non-blocking saver for dataset capture (Option A)
# ---------------------------------------------------------
# pip install ultralytics opencv-python pillow imagehash
from ultralytics import YOLO
import cv2, time, threading, queue, argparse
import numpy as np
from pathlib import Path
from collections import deque
from PIL import Image
import imagehash

# ---------- Config ----------
MODEL_WEIGHTS = 'yolov8n.pt'       # or 'yolov8s.pt' for better accuracy
CAM_INDEX = 0
TRAIN_DIR = Path("training_data")
TEST_DIR = Path("testing_data")
QUEUE_MAXSIZE = 64                 # drop newest when full (to avoid blocking UI)
MIN_SAVE_INTERVAL = 0.8            # seconds between saves
PHASH_DISTANCE_THRESHOLD = 12      # 10..16 (higher = stricter dedup)
BLUR_VAR_THR = 90.0                # variance of Laplacian threshold
ROI_PIX_FRAC_THR = 0.06            # fraction of ROI pixels that must change
ROI_DIFF_THR = 15                  # pixel diff threshold (0..255)
MIN_SHIFT_FRAC = 0.10              # bbox center shift / diag
MIN_SCALE_FRAC = 0.18              # bbox area change
MIN_ASPECT_DELTA = 0.10            # aspect-ratio relative change

# ---------- Helpers ----------
def crop_frame(frame_bgr, box):
    """box is an Ultralytics Boxes.xyxy tensor-like (x1,y1,x2,y2)."""
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(int, np.clip(np.array(box, dtype=np.int32),
                                      [0, 0, 0, 0], [w-1, h-1, w-1, h-1]))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2].copy()

def is_blurry(img_bgr, thr=BLUR_VAR_THR):
    return cv2.Laplacian(img_bgr, cv2.CV_64F).var() < thr

def roi_motion_ok(curr_gray, last_gray,
                  pix_frac_thr=ROI_PIX_FRAC_THR, diff_thr=ROI_DIFF_THR):
    if last_gray is None or curr_gray.shape != last_gray.shape:
        return True
    diff = cv2.absdiff(curr_gray, last_gray)
    changed_frac = (diff > diff_thr).mean()
    return changed_frac >= pix_frac_thr

def is_diverse_box(x1,y1,x2,y2, prev_box,
                   min_shift_frac=MIN_SHIFT_FRAC,
                   min_scale_frac=MIN_SCALE_FRAC,
                   min_aspect_delta=MIN_ASPECT_DELTA):
    if prev_box is None:
        return True
    px1,py1,px2,py2 = prev_box
    w,h = x2-x1, y2-y1
    pw,ph = px2-px1, py2-py1
    if w<=0 or h<=0 or pw<=0 or ph<=0:
        return True
    cx,cy = (x1+x2)/2, (y1+y2)/2
    pcx,pcy = (px1+px2)/2, (py1+py2)/2
    diag = max(1.0, np.hypot(w,h))
    pos_shift = np.hypot(cx-pcx, cy-pcy) / diag
    scale_change = abs((w*h) - (pw*ph)) / max(1.0, pw*ph)
    aspect_change = abs((w/h) - (pw/ph)) / max(1e-6, (pw/ph))
    return (pos_shift > min_shift_frac) or (scale_change > min_scale_frac) or (aspect_change > min_aspect_delta)

# ---------- Background saver thread ----------
def make_saver(save_dir: Path):
    recent_hashes = deque(maxlen=300)
    last_save_time = [0.0]           # boxed for closure write
    last_roi_gray  = [None]
    last_box_saved = [None]

    def save_worker(q: "queue.Queue[dict]"):
        while True:
            item = q.get()
            if item is None:     # shutdown
                q.task_done()
                break

            cut, frame_id, xyxy = item["crop"], item["frame_id"], item["xyxy"]
            now = time.time()

            # 1) time throttle
            if now - last_save_time[0] < MIN_SAVE_INTERVAL:
                q.task_done()
                continue

            # 2) quality
            if is_blurry(cut):
                q.task_done()
                continue

            # 3) ROI motion
            roi_gray = cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
            if not roi_motion_ok(roi_gray, last_roi_gray[0]):
                q.task_done()
                continue

            # 4) bbox diversity vs last saved
            x1,y1,x2,y2 = xyxy
            if not is_diverse_box(x1,y1,x2,y2, last_box_saved[0]):
                q.task_done()
                continue

            # 5) pHash dedup
            img_pil = Image.fromarray(cv2.cvtColor(cut, cv2.COLOR_BGR2RGB))
            ph = imagehash.phash(img_pil, hash_size=16)
            if any((ph - h) < PHASH_DISTANCE_THRESHOLD for h in recent_hashes):
                q.task_done()
                continue

            # Passed all gates → save
            ts = int(now)
            out = save_dir / f"{ts}_{frame_id:06d}.png"
            img_pil.save(out)

            # update state
            recent_hashes.append(ph)
            last_save_time[0]  = now
            last_roi_gray[0]   = roi_gray
            last_box_saved[0]  = (x1,y1,x2,y2)
            q.task_done()

    return save_worker

# ---------- Main ----------
def main():
    # --- CLI ---
    parser = argparse.ArgumentParser(description="Collect person crops into train or test folders")
    parser.add_argument('-m', '--mode', choices=['train', 'test'], default='train',
                        help="choose whether collected images go to 'train' or 'test' folder")
    parser.add_argument('-o', '--out', type=str, default='trento_house_data',
                        help="parent data folder (images will go to [out]/train or [out]/test)")
    args = parser.parse_args()

    # Build save directory: [data_name]/train or [data_name]/test
    data_parent = Path(args.out)
    if args.mode == 'train':
        SAVE_DIR = data_parent / "train"
    else:
        # For test mode, save to [data_name]/test/normal (anomalous handled separately)
        SAVE_DIR = data_parent / "test" / "normal"
    
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving captures to: {SAVE_DIR} (mode={args.mode})")

    # Load YOLO in main thread (inference stays here to keep preview immediate)
    model = YOLO(MODEL_WEIGHTS)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return

    # Start background saver
    save_queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
    saver_thread = threading.Thread(target=make_saver(SAVE_DIR), args=(save_queue,), daemon=True)
    saver_thread.start()

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO inference (fast path)
            results = model(frame, stream=True)

            # Draw + enqueue crops to saver without blocking UI
            for r in results:
                if r.boxes is None: 
                    continue
                names = r.names
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if names[cls] != 'person':
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])

                    # show boxes
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, f'person {conf:.2f}', (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    # prepare crop for saver
                    crop = crop_frame(frame, (x1,y1,x2,y2))
                    if crop is None or crop.size == 0:
                        continue

                    # non-blocking enqueue: drop if queue is full
                    if not save_queue.full():
                        save_queue.put_nowait({"crop": crop, "frame_id": frame_count, "xyxy": (x1,y1,x2,y2)})

            cv2.imshow(f"YOLOv8 Person Capture (mode={args.mode})", frame)
            key = cv2.waitKey(1)
            if key == 27:    # ESC
                break

            frame_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        # graceful shutdown of saver
        try:
            save_queue.put_nowait(None)
        except queue.Full:
            # ensure space then send sentinel
            _ = save_queue.get_nowait()
            save_queue.task_done()
            save_queue.put_nowait(None)

        save_queue.join()  # wait until all tasks done
        # (daemon thread will exit after sentinel)

if __name__ == "__main__":
    main()