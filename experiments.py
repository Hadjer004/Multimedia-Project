import matplotlib.pyplot as plt
from Intra_frame import run_intra_experiment
from Inter_frame import run_inter_experiment

# =========================
# 1. INTRA EXPERIMENT
# =========================

fq_values = [5, 10, 15, 20, 30]
intra_ratios = []

print("\nRunning Intra-frame experiments...\n")

for fq in fq_values:
    ratio = run_intra_experiment(fq)
    intra_ratios.append(ratio)
    print(f"fq={fq} -> ratio={ratio:.2f}")


plt.figure()
plt.plot(fq_values, intra_ratios, marker='o')
plt.title("Compression Ratio vs Quantization Factor")
plt.xlabel("Quantization Factor")
plt.ylabel("Compression Ratio")
plt.grid()
plt.savefig("fq_vs_compression.png")


# =========================
# 2. INTER EXPERIMENT (GOP)
# =========================

gop_values = [1, 5, 10, 15]
gop_ratios = []

print("\nRunning Inter-frame experiments...\n")

for gop in gop_values:
    ratio = run_inter_experiment(gop)
    gop_ratios.append(ratio)
    print(f"GOP={gop} -> ratio={ratio:.2f}")


plt.figure()
plt.plot(gop_values, gop_ratios, marker='o')
plt.title("GOP Size vs Compression Ratio")
plt.xlabel("GOP Size")
plt.ylabel("Compression Ratio")
plt.grid()
plt.savefig("gop_vs_compression.png")

print("\nExperiments completed. Graphs saved.")