import pickle
import zlib


def encode_entropy(sequence, output_file="output.bin"):

    data = pickle.dumps(sequence)
    compressed = zlib.compress(data, level=9)

    with open(output_file, "wb") as f:
        f.write(compressed)

    print("Entropy encoding done")
    return output_file


def decode_entropy(input_file="output.bin"):

    with open(input_file, "rb") as f:
        compressed = f.read()

    data = zlib.decompress(compressed)
    sequence = pickle.loads(data)

    print("Entropy decoding done")
    return sequence


if __name__ == "__main__":
    pass