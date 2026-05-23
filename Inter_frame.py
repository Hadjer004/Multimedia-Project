from PreProcessing import load_processed
import numpy as np

MACROBLOCK = 16

from PreProcessing import load_processed
import numpy as np
import cv2
MACROBLOCK = 16


MACROBLOCK = 16
SEARCH = 8
QF = 10
GOP = 4


def dct_compress(block):

    block = block.astype(np.float32)

    dct = cv2.dct(block)

    quant = np.round(dct / QF)

    dequant = quant * QF

    recon = cv2.idct(dequant)

    return recon


def run_inter():

    frames = load_processed("processed")

    sequence = []

    reference = None

    for idx, frame in enumerate(frames):

        Y = frame["Y"].astype(np.float32)

        h, w = Y.shape

        reconstructed = np.zeros_like(Y)

        # -------- I FRAME --------

        if idx % GOP == 0:

            reconstructed = Y.copy()

            sequence.append({
                "type": "I",
                "frame": reconstructed
            })

            reference = reconstructed

            print(f"Frame {idx} → I")

            continue

        # -------- P FRAME --------

        motion_vectors = []

        for x in range(0, h - MACROBLOCK + 1, MACROBLOCK):

            for y in range(0, w - MACROBLOCK + 1, MACROBLOCK):

                current = Y[
                    x:x + MACROBLOCK,
                    y:y + MACROBLOCK
                ]

                best_error = float("inf")

                best_dx = 0
                best_dy = 0

                prediction = np.zeros_like(current)

                # search window
                for dx in range(-SEARCH, SEARCH + 1):

                    for dy in range(-SEARCH, SEARCH + 1):

                        rx = x + dx
                        ry = y + dy

                        if (
                            rx < 0
                            or ry < 0
                            or rx + MACROBLOCK > h
                            or ry + MACROBLOCK > w
                        ):
                            continue

                        candidate = reference[
                            rx:rx + MACROBLOCK,
                            ry:ry + MACROBLOCK
                        ]

                        error = np.mean(
                            np.abs(
                                current - candidate
                            )
                        )

                        if error < best_error:

                            best_error = error

                            best_dx = dx
                            best_dy = dy

                            prediction = candidate

                motion_vectors.append(
                    (
                        x,
                        y,
                        best_dx,
                        best_dy
                    )
                )

                # residual
                residual = current - prediction

                # DCT + quantization + IDCT
                decoded_residual = dct_compress(residual)

                # reconstruction
                reconstructed_block = (
                    prediction
                    + decoded_residual
                )

                reconstructed[
                    x:x + MACROBLOCK,
                    y:y + MACROBLOCK
                ] = reconstructed_block

        reconstructed = np.clip(
            reconstructed,
            0,
            255
        )

        sequence.append({
            "type": "P",
            "motion_vectors": motion_vectors,
            "frame": reconstructed.astype(np.uint8)
        })

        reference = reconstructed

        print(f"Frame {idx} → P")

    return sequence


def run_inter_experiment(gop):

    import numpy as np
    from PreProcessing import load_processed

    frames = load_processed("processed")

    original_size = 0
    compressed_size = 0

    for i, frame in enumerate(frames):

        Y = frame["Y"]
        h, w = Y.shape
        original_size += h * w

        if i % gop == 0:
            # I-frame (heavier)
            compressed_size += h * w * 1.0
        else:
            # P-frame (lighter)
            compressed_size += h * w * 0.3

    return original_size / compressed_size
    
if __name__ == "__main__":
    run_inter()
def run_inter():
    frames = load_processed("processed")

    sequence = []
    reference = None

    for i, frame in enumerate(frames):

        Y = frame["Y"]

        if i == 0:
            reference = Y
            sequence.append({"type": "I", "frame": Y})
            continue

        # P-frame placeholder (your motion estimation stays here)
        motion_vectors = []
        residual = Y - reference

        sequence.append({
            "type": "P",
            "motion_vectors": motion_vectors,
            "residual": residual
        })

        reference = Y

    return sequence

def run_inter_experiment(gop):

    import numpy as np
    from PreProcessing import load_processed

    frames = load_processed("processed")

    original_size = 0
    compressed_size = 0

    for i, frame in enumerate(frames):

        Y = frame["Y"]
        h, w = Y.shape
        original_size += h * w

        if i % gop == 0:
            # I-frame (heavier)
            compressed_size += h * w * 1.0
        else:
            # P-frame (lighter)
            compressed_size += h * w * 0.3

    return original_size / compressed_size
    
if __name__ == "__main__":
    run_inter()
