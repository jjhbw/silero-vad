# Silero VAD Node Fork

Minimal Node.js wrapper around the Silero VAD ONNX model, with a small CLI and parity tests against the Python implementation. The Node implementation runs VAD and silence stripping directly from ffmpeg streams to keep memory usage low on long files.

## Install

```bash
npm install @jjhbw/silero-vad
```

Requires Node 18+ and `ffmpeg` available on `PATH` for decoding arbitrary audio formats.

## Library usage

```js
const {
  loadSileroVad,
  getSpeechTimestamps,
  writeStrippedAudio,
  WEIGHTS,
} = require("@jjhbw/silero-vad");

(async () => {
  const vad = await loadSileroVad("default"); // or WEIGHTS keys/custom path
  try {
    if (!vad.sampleRate) throw new Error("Model sample rate is undefined");
    const inputs = ["input.wav", "other.mp3"];
    for (const inputPath of inputs) {
      vad.resetStates(); // per file/stream
      const ts = await getSpeechTimestamps(inputPath, vad, {
        returnSeconds: true,
      });
      // Each entry includes both seconds (start/end) and samples (startSample/endSample).
      console.log(inputPath, ts);
      // Example return value:
      // [
      //   { start: 0.36, end: 1.92, startSample: 5760, endSample: 30720 },
      //   { start: 2.41, end: 3.05, startSample: 38560, endSample: 48800 }
      // ]

      // Strip silences from the original file using the timestamps.
      // Pick any extension supported by ffmpeg (e.g., .wav, .flac).
      // Note: encoding speed varies by container/codec; uncompressed PCM (e.g., .wav) is fastest,
      // lossless compression (e.g., .flac) is slower, and lossy codecs (e.g., .mp3/.aac/.opus)
      // are typically the slowest to encode.
      const outPath = inputPath.replace(/\.[^.]+$/, ".stripped.wav");
      await writeStrippedAudio(inputPath, ts, vad.sampleRate, outPath);
    }
  } finally {
    await vad.session.release?.(); // once per process when shutting down
  }
})();
```

Guidelines:

- Load once, reuse: keep one `SileroVad` per concurrent worker.
- Call `resetStates()` before each new file/stream; the session and weights stay in memory.
- Call `release()` when shutting down.

## CLI

```bash
npx @jjhbw/silero-vad --audio input.wav --audio other.mp3 [options]
```

Options:

- `--model <key|path>`: model key (`default`, `16k`, `8k_16k`, `half`, `op18`) or custom ONNX path (default: `default`, i.e., bundled 16k op15).
- `--threshold <float>`: speech probability threshold (default `0.5`).
- `--min-speech-ms <ms>`: minimum speech duration in ms (default `250`).
- `--min-silence-ms <ms>`: minimum silence duration in ms (default `100`).
- `--speech-pad-ms <ms>`: padding added to each speech segment in ms (default `30`).
- `--time-resolution <n>`: decimal places for seconds output (default `3`).
- `--neg-threshold <float>`: override the negative threshold (default `max(threshold - 0.15, 0.01)`).
- `--seconds`: output timestamps in seconds (default on).
- `--cps <float>`: enable the timeline visualization and set chars per second (default `4`).
- `--strip-silence`: write a new WAV file with silences removed.
- `--output-dir <path>`: output directory for strip-silence files (default: input dir).

Outputs an array of `{ file, timestamps }` to stdout as JSON. The CLI reuses a single ONNX session and resets state per file.
The sample rate is defined by the selected model (read from `vad.sampleRate`).

## Development

Clone the repo to run benchmarks and tests locally.

### Benchmark

```bash
git clone https://github.com/jjhbw/silero-vad
cd silero-vad/js-fork
npm install
node bench.js --audio data/test.mp3 --runs 5
```

The benchmark reports timings per file for streaming VAD and silence stripping. Stripped-audio files are written to a temporary directory and removed after each run.

### Tests

Snapshot tests compare Node outputs against Python ground truth (`tests/snapshots/onnx.json`):

```bash
git clone https://github.com/jjhbw/silero-vad
cd silero-vad/js-fork
npm install
npm test
```

Ensure Python snapshots are generated (run `pytest tests/test_snapshots.py` in the repo root) and `ffmpeg` is installed.
