"""
Interactive labeling helper.
Usage: python label_helper.py data/raw/run02.mp4

Controls:
  d / right arrow -> next frame
  a / left arrow  -> previous frame
  SPACE           -> mark current frame as the drop frame and quit
  n               -> mark this whole video as "normal" and quit (no drop frame)
  q               -> quit without saving

Appends one row to data/labels.csv each time you run it on a video.
"""
import cv2
import sys
import csv
import os

LABELS_CSV = "data/labels.csv"

def main(video_path):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    def show(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            return None
        label = f"frame {idx}/{total}  ({idx/fps:.2f}s)"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        cv2.imshow(video_id, frame)
        return frame

    show(frame_idx)
    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('d'), 83):          # next frame
            frame_idx = min(frame_idx + 1, total - 1)
            show(frame_idx)
        elif key in (ord('a'), 81):        # previous frame
            frame_idx = max(frame_idx - 1, 0)
            show(frame_idx)
        elif key == ord(' '):              # mark drop frame here
            save_row(video_id, "failure", frame_idx)
            print(f"Saved: {video_id} = failure, drop_frame={frame_idx}")
            break
        elif key == ord('n'):              # mark as normal
            save_row(video_id, "normal", "")
            print(f"Saved: {video_id} = normal")
            break
        elif key == ord('q'):
            print("Quit without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()

def save_row(video_id, vtype, drop_frame):
    file_exists = os.path.isfile(LABELS_CSV)
    os.makedirs(os.path.dirname(LABELS_CSV), exist_ok=True)
    with open(LABELS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["video_id", "type", "drop_frame"])
        writer.writerow([video_id, vtype, drop_frame])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python label_helper.py <path_to_video>")
        sys.exit(1)
    main(sys.argv[1])