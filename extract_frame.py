import cv2
import os

VIDEO_PATH = "video - Trim.mp4"
OUTPUT_DIR = "frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)

frame_count = 0

while True:

    # Read frame
    ret, frame = cap.read()

    # Stop if video ended
    if not ret:
        break

    # Save frame
    frame_path = os.path.join(
        OUTPUT_DIR,
        f"frame_{frame_count:04d}.png"
    )

    cv2.imwrite(frame_path, frame)

    print(f"Saved frame {frame_count}")

    frame_count += 1

cap.release()

print("Frame extraction completed")