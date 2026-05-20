from PreProcessing import load_processed
import numpy as np

MACROBLOCK = 16


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