import json
from pathlib import Path

import pytest
import torch
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

torch.set_num_threads(1)

SAMPLES = [
    "tests/data/test.wav",
    "tests/data/test.opus",
    "tests/data/test.mp3",
]
SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _compute_snapshot(model, sample_paths, sampling_rate=16000):
    results = []
    for sample in sample_paths:
        audio = read_audio(sample, sampling_rate=sampling_rate)
        ts = get_speech_timestamps(
            audio,
            model,
            visualize_probs=False,
            return_seconds=True,
            time_resolution=3,
        )
        results.append(
            {
                "file": Path(sample).name,
                "sampling_rate": sampling_rate,
                "speech_timestamps": ts,
            }
        )
    return results


def _write_or_assert(snapshot_path: Path, data):
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    if snapshot_path.exists():
        with snapshot_path.open() as f:
            expected = json.load(f)
        assert data == expected
    else:
        with snapshot_path.open("w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
        pytest.fail(f"Snapshot created at {snapshot_path}. Verify and re-run.")


def test_jit_snapshot():
    model = load_silero_vad(onnx=False)
    data = {
        "model": "jit",
        "snapshots": _compute_snapshot(model, SAMPLES),
    }
    _write_or_assert(SNAPSHOT_DIR / "jit.json", data)


def test_onnx_snapshot():
    model = load_silero_vad(onnx=True)
    data = {
        "model": "onnx",
        "snapshots": _compute_snapshot(model, SAMPLES),
    }
    _write_or_assert(SNAPSHOT_DIR / "onnx.json", data)
