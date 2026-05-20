import cv2
import numpy as np
from entropy_encoder import decode_entropy

sequence = decode_entropy("output.bin")

print("Entropy decoding done")

# find size
for item in sequence:
    if item["type"] == "I":
        h, w = item["frame"].shape
        break

out = cv2.VideoWriter(
    "reconstructed.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (w, h),
    isColor=False
)

reference = None

for item in sequence:

    # ---------------- I FRAME ----------------
    if item["type"] == "I":
        frame = item["frame"]
        reference = frame

    # ---------------- P FRAME ----------------
    else:
        frame = reference + item["residual"]
        reference = frame

    frame = np.clip(frame, 0, 255).astype(np.uint8)

    out.write(frame)

out.release()

print("Video reconstructed ✔")