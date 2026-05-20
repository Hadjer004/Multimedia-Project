# Multimedia-Project
A simplified MPEG-style video compression pipeline implemented in Python using spatial and temporal compression techniques.

## 1. Project Structure
project/
│
├── frames/                     # Extracted original frames
├── processed/                  # Preprocessed Y/Cb/Cr data
├── reconstructed_frames/       # Reconstructed frames
│
├── extract_frames.py
├── PreProcessing.py
├── Intra_frame.py
├── Inter_frame.py
├── entropy_encoder.py
├── reconstruct_video.py
├── experiments.py
├── main.py
│
├── output.bin                  # Final compressed file
├── reconstructed.mp4           # Reconstructed video
├── video - Trim.mp4            # original video
└── README.md

## 2.Technologies Used
Python, OpenCV, NumPy, Matplotlib, zlib, pickle

## 3.How to Run
python main.py

## 4.Educational Objective
The goal of this project is to understand the fundamental principles of video compression systems by reproducing a simplified MPEG-like pipeline using Python.
