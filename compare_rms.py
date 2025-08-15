import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Plot the RMS from two audio files


def compute_rms(audio_path, frame_length=2048, hop_length=512):
    y, sr = librosa.load(audio_path, sr=None)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return times, rms


def plot_rms(audio_path1, audio_path2):
    times1, rms1 = compute_rms(audio_path1)
    times2, rms2 = compute_rms(audio_path2)

    plt.figure(figsize=(12, 6))
    plt.plot(times1, rms1, label=f"RMS: {audio_path1}", alpha=0.8)
    plt.plot(times2, rms2, label=f"RMS: {audio_path2}", alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("RMS Energy")
    plt.title("RMS Energy Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot RMS energy of two audio files.")
    parser.add_argument("audio1", type=str, help="Path to the first audio file")
    parser.add_argument("audio2", type=str, help="Path to the second audio file")
    args = parser.parse_args()

    plot_rms(args.audio1, args.audio2)


if __name__ == "__main__":
    main()
