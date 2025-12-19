# Silero VAD Node Fork

Minimal Node.js wrapper around the Silero VAD ONNX model, with a small CLI and parity tests against the Python implementation.

## Install

```bash
cd js-fork
npm install
```

Requires Node 18+ and `ffmpeg` available on `PATH` for decoding arbitrary audio formats.

## CLI

```bash
npx silero-vad-cli --audio input.wav --audio other.mp3 [options]
```

Options:
- `--model <key|path>`: model key (`default`, `16k`, `8k_16k`, `half`, `op18`) or custom ONNX path (default: `default`, i.e., bundled 16k op15).
- `--threshold <float>`: speech probability threshold (default `0.5`).
- `--min-speech-ms <ms>`: minimum speech duration in ms (default `250`).
- `--min-silence-ms <ms>`: minimum silence duration in ms (default `100`).
- `--speech-pad-ms <ms>`: padding added to each speech segment in ms (default `30`).
- `--time-resolution <n>`: decimal places for seconds output (default `3`).
- `--neg-threshold <float>`: override the negative threshold (default `threshold - 0.15`).
- `--seconds`: output timestamps in seconds (default on).

Outputs an array of `{ file, timestamps }` to stdout as JSON. The CLI reuses a single ONNX session and resets state per file.
The sample rate is defined by the selected model (read from `vad.sampleRate`); it is not configurable in `getSpeechTimestamps`.

## Library usage

```js
const {
  loadSileroVad,
  decodeWithFfmpeg,
  getSpeechTimestamps,
  WEIGHTS,
} = require('./lib');

(async () => {
  const vad = await loadSileroVad('default'); // or WEIGHTS keys/custom path
  try {
    if (!vad.sampleRate) throw new Error('Model sample rate is undefined');
    const sr = vad.sampleRate;
    vad.resetStates(); // per file/stream
    const audio = await decodeWithFfmpeg('input.wav', { sampleRate: sr });
    const ts = await getSpeechTimestamps(audio, vad, { returnSeconds: true });
    // Each entry includes both seconds (start/end) and samples (startSample/endSample).
    console.log(ts);
  } finally {
    await vad.session.release?.();
  }
})();
```

Guidelines:
- Load once, reuse: keep one `SileroVad` per concurrent worker.
- Call `resetStates()` before each new file/stream; the session and weights stay in memory.
- Call `release()` when shutting down.

## Tests

Snapshot tests compare Node outputs against Python ground truth (`tests/snapshots/onnx.json`):

```bash
cd js-fork
npm test
```

Ensure Python snapshots are generated (run `pytest tests/test_snapshots.py` in the repo root) and `ffmpeg` is installed.***
