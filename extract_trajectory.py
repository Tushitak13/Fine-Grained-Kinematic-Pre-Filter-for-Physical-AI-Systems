import os
import csv
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

RAW_DIR = "data/raw"
LABELS_CSV = "data/labels.csv"
OUTPUT_DIR = "data/trajectories"


def extract_trajectory_from_video(video_path):
    """
    Opens a video file using OpenCV and uses MediaPipe to extract
    a sequence of 63-float landmark arrays (21 landmarks x 3 coords).
    Returns (trajectory, fps).
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    trajectory = []
    last_known_frame = np.zeros(63, dtype=np.float32)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            frame_array = np.array(coords, dtype=np.float32)
            last_known_frame = frame_array
        else:
            frame_array = last_known_frame

        trajectory.append(frame_array)

    cap.release()
    hands.close()
    return np.array(trajectory), fps


def batch_process_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(LABELS_CSV):
        print(f"{LABELS_CSV} not found. Run label_helper.py on your videos first.")
        return

    with open(LABELS_CSV) as f:
        rows = list(csv.DictReader(f))

    print(f"Found {len(rows)} labeled videos in {LABELS_CSV}...")

    for row in rows:
        video_id = row["video_id"]
        video_path = None
        for ext in (".mp4", ".avi", ".mov"):
            candidate = os.path.join(RAW_DIR, video_id + ext)
            if os.path.exists(candidate):
                video_path = candidate
                break

        if video_path is None:
            print(f"Skipping {video_id}: no matching file in {RAW_DIR}")
            continue

        trajectory, fps = extract_trajectory_from_video(video_path)

        if len(trajectory) < 10:
            print(f"Skipping {video_id}: too short or hand not detected.")
            continue

        save_path = os.path.join(OUTPUT_DIR, video_id + ".npy")
        np.save(save_path, trajectory)

        # save fps alongside — Person 2 needs it to convert drop_frame to seconds
        fps_path = os.path.join(OUTPUT_DIR, video_id + "_fps.txt")
        with open(fps_path, "w") as f:
            f.write(str(fps))

        print(f"Processed {video_id} ({row['type']}) -> {save_path}  shape={trajectory.shape}  fps={fps:.1f}")


if __name__ == "__main__":
    batch_process_all()