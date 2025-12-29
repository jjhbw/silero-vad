#!/usr/bin/env python3
import argparse
import statistics
import time

from silero_vad import get_speech_timestamps, load_silero_vad, read_audio


def parse_args():
    parser = argparse.ArgumentParser(
        description="One-off benchmark for Python get_speech_timestamps."
    )
    parser.add_argument(
        "--audio",
        nargs="+",
        required=True,
        help="One or more audio file paths to benchmark.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of timed runs per file (default: 5).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs per file (default: 1).",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Sampling rate for read_audio/get_speech_timestamps (default: 16000).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Speech threshold passed to get_speech_timestamps (default: 0.5).",
    )
    parser.add_argument(
        "--onnx",
        action="store_true",
        help="Use the ONNX model for benchmarking.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=16,
        help="ONNX opset version when --onnx is set (default: 16).",
    )
    parser.add_argument(
        "--return-seconds",
        action="store_true",
        help="Return timestamps in seconds instead of samples.",
    )
    return parser.parse_args()


def format_seconds(seconds):
    return f"{seconds:.4f}s"


def main():
    args = parse_args()
    model = load_silero_vad(onnx=args.onnx, opset_version=args.opset)

    for path in args.audio:
        audio = read_audio(path, sampling_rate=args.sampling_rate)
        audio_duration = audio.numel() / float(args.sampling_rate)

        for _ in range(args.warmup):
            get_speech_timestamps(
                audio,
                model,
                threshold=args.threshold,
                sampling_rate=args.sampling_rate,
                return_seconds=args.return_seconds,
            )

        timings = []
        for _ in range(args.runs):
            start = time.perf_counter()
            get_speech_timestamps(
                audio,
                model,
                threshold=args.threshold,
                sampling_rate=args.sampling_rate,
                return_seconds=args.return_seconds,
            )
            timings.append(time.perf_counter() - start)

        mean_time = statistics.mean(timings)
        min_time = min(timings)
        max_time = max(timings)
        rtf = mean_time / audio_duration if audio_duration else float("inf")

        print(f"{path}:")
        print(f"  duration: {format_seconds(audio_duration)}")
        print(f"  runs: {args.runs}, warmup: {args.warmup}")
        print(
            f"  mean: {format_seconds(mean_time)} | "
            f"min: {format_seconds(min_time)} | "
            f"max: {format_seconds(max_time)} | "
            f"rtf: {rtf:.3f}x"
        )


if __name__ == "__main__":
    main()
