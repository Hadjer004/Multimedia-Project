import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PreProcessing import preprocess
from Intra_frame import run_intra
from Inter_frame import run_inter
from entropy_encoder import encode_entropy, decode_entropy


# =========================
# STEP 1 — PREPROCESSING
# =========================
print("\n[1] Preprocessing video frames...")
preprocess()
print("[1] Done\n")


# =========================
# STEP 2 — INTRA FRAME TEST (I-FRAMES)
# =========================
print("[2] Running intra-frame coding...")
intra_frames = run_intra()
print("[2] Done\n")


# =========================
# STEP 3 — INTER FRAME (P-FRAMES)
# =========================
print("[3] Running inter-frame coding...")
sequence = run_inter()
print("[3] Done\n")


# =========================
# STEP 4 — ENTROPY ENCODING (.bin)
# =========================
print("[4.a] Entropy encoding (compression)...")
encode_entropy(sequence, output_file="output.bin")
print("[4.a] Done\n")


# =========================
# STEP 5 — ENTROPY DECODING (TEST)
# =========================
print("[4.b] Decoding entropy file...")
decoded_sequence = decode_entropy("output.bin")
print("[4.b] Done\n")


# =========================
# FINAL
# =========================
print("✅ FULL VIDEO COMPRESSION PIPELINE COMPLETED SUCCESSFULLY")

print("\n========== STEP 5A: QUALITY METRICS ==========\n")

# -----------------------------
# 1. COMPRESSION RATIO
# -----------------------------
original_size = 0

for file in os.listdir("frames"):
    original_size += os.path.getsize(os.path.join("frames", file))

compressed_size = os.path.getsize("output.bin")

compression_ratio = original_size / compressed_size

print("Original size (bytes):", original_size)
print("Compressed size (bytes):", compressed_size)
print("Compression ratio:", round(compression_ratio, 2))


# -----------------------------
# 2. FRAME TYPE BREAKDOWN
# -----------------------------
sequence = decode_entropy("output.bin")

I_frames = 0
P_frames = 0

for item in sequence:
    if item["type"] == "I":
        I_frames += 1
    else:
        P_frames += 1

print("\nFrame breakdown:")
print("I-frames:", I_frames)
print("P-frames:", P_frames)

print("\n========== END OF STEP 5A ==========\n")

print("\n========== STEP 5B: VISUALISATION ==========\n")

# -----------------------------
# 1. ORIGINAL FRAMES
# -----------------------------
files = sorted(os.listdir("frames"))

plt.figure(figsize=(10,4))

for i in range(3):
    img = cv2.imread(os.path.join("frames", files[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(1,3,i+1)
    plt.imshow(img)
    plt.title(f"Frame {i}")
    plt.axis("off")

plt.suptitle("Original Frames")
plt.show()


# -----------------------------
# 2. Y, Cb, Cr CHANNELS
# -----------------------------
img = cv2.imread(os.path.join("frames", files[0]))
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

Y, Cr, Cb = cv2.split(ycrcb)

plt.figure(figsize=(10,3))

plt.subplot(1,3,1)
plt.imshow(Y, cmap="gray")
plt.title("Y (Luminance)")

plt.subplot(1,3,2)
plt.imshow(Cr, cmap="gray")
plt.title("Cr")

plt.subplot(1,3,3)
plt.imshow(Cb, cmap="gray")
plt.title("Cb")

plt.suptitle("Color Space Conversion")
plt.show()


# -----------------------------
# 3. DCT PIPELINE (8x8 BLOCK)
# -----------------------------
Q = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.float32)

img = cv2.imread(os.path.join("frames", files[0]), 0)

block = img[0:8, 0:8].astype(np.float32) - 128

dct = cv2.dct(block)

quant = np.round(dct / Q)

dequant = quant * Q

recon = cv2.idct(dequant) + 128

plt.figure(figsize=(10,3))

plt.subplot(1,4,1)
plt.imshow(block + 128, cmap="gray")
plt.title("Original Block")

plt.subplot(1,4,2)
plt.imshow(dct, cmap="gray")
plt.title("DCT")

plt.subplot(1,4,3)
plt.imshow(quant, cmap="gray")
plt.title("Quantized")

plt.subplot(1,4,4)
plt.imshow(recon, cmap="gray")
plt.title("Reconstructed")

plt.suptitle("DCT Pipeline (8x8 Block)")
plt.show()

print("========== END OF STEP 5B ==========\n")