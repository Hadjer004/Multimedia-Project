import os
import cv2
import numpy as np

INPUT_DIR = "frames"
OUTPUT_DIR = "processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess():
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".png")])

    for i, file in enumerate(files):
        path = os.path.join(INPUT_DIR, file)
        frame = cv2.imread(path)

        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycrcb)

        Cb = cv2.resize(Cb, (Cb.shape[1] // 2, Cb.shape[0] // 2))
        Cr = cv2.resize(Cr, (Cr.shape[1] // 2, Cr.shape[0] // 2))

        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f"frame_{i:04d}.npz"),
            Y=Y, Cb=Cb, Cr=Cr
        )

    print("Preprocessing done")


def load_processed(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npz")])

    frames = []
    for f in files:
        data = np.load(os.path.join(folder, f))
        frames.append({"Y": data["Y"], "Cb": data["Cb"], "Cr": data["Cr"]})

    return frames


if __name__ == "__main__":
    preprocess()