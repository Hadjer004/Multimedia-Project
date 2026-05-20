import os
import cv2
import numpy as np
from PreProcessing import load_processed

OUTPUT_DIR = "reconstructed_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_intra():
    processed = load_processed("processed")

    reconstructed_frames = []

    for i, frame in enumerate(processed):

        Y = frame["Y"]

        # (your DCT logic goes here)
        reconstructed = Y.copy()  # replace later with real DCT result

        reconstructed_frames.append(reconstructed)

        # SAVE FRAME
        output_path = os.path.join(OUTPUT_DIR, f"frame_{i:04d}.png")
        cv2.imwrite(output_path, reconstructed)

        print(f"Frame {i} reconstructed")

    return reconstructed_frames

def run_intra_experiment(fq):
    import os
    import cv2
    import numpy as np
    from PreProcessing import load_processed

    INPUT_DIR = "processed"

    processed = load_processed(INPUT_DIR)

    original_size = 0
    compressed_size = 0

    for frame in processed:

        Y = frame["Y"]

        h, w = Y.shape
        original_size += h * w

        # --- compression ---
        # same logic you already have but simplified:

        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8

        padded = np.pad(Y, ((0, pad_h), (0, pad_w)), mode='constant')
        padded = padded.astype(np.float32)

        for i in range(0, padded.shape[0], 8):
            for j in range(0, padded.shape[1], 8):

                block = padded[i:i+8, j:j+8] - 128
                dct = cv2.dct(block)

                quant = np.round(dct / fq)

                non_zero = np.count_nonzero(quant)
                compressed_size += non_zero  # approximation

    return original_size / compressed_size
if __name__ == "__main__":
    run_intra()