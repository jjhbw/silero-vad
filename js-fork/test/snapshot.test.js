const assert = require('assert');
const fs = require('fs');
const path = require('path');
const test = require('node:test');

const { getSpeechTimestamps, loadSileroVad } = require('..');

const ROOT = path.join(__dirname, '..', '..');
const SNAPSHOT_DIR = path.join(ROOT, 'tests', 'snapshots');
const DATA_DIR = path.join(ROOT, 'tests', 'data');

function readSnapshot(name) {
  const p = path.join(SNAPSHOT_DIR, name);
  const raw = fs.readFileSync(p, 'utf8');
  return JSON.parse(raw);
}

function assertTimestampsClose(actual, expected, tolerance, label) {
  assert.strictEqual(actual.length, expected.length, `${label} length mismatch`);
  for (let i = 0; i < actual.length; i += 1) {
    const a = actual[i];
    const e = expected[i];
    assert.ok(Math.abs(a.start - e.start) <= tolerance, `${label} start mismatch at ${i}`);
    assert.ok(Math.abs(a.end - e.end) <= tolerance, `${label} end mismatch at ${i}`);
  }
}

test('onnx snapshot matches python ground truth', async () => {
  const snapshot = readSnapshot('onnx.json');
  const vad = await loadSileroVad('default');

  for (const entry of snapshot.snapshots) {
    await test(`file ${entry.file}`, async () => {
      vad.resetStates();
      const wavPath = path.join(DATA_DIR, entry.file);
      const ts = await getSpeechTimestamps(wavPath, vad, {
        threshold: 0.5,
        returnSeconds: true,
        timeResolution: 3,
      });

      const plainTs = ts.map(({ start, end }) => ({ start, end }));

      if (entry.file === 'test.mp3') {
        // MP3 decoding in Node (ffmpeg) vs Python (torchaudio) can differ by encoder delay/padding,
        // so allow a small tolerance instead of strict equality.
        assertTimestampsClose(plainTs, entry.speech_timestamps, 0.1, entry.file);
      } else {
        assert.deepStrictEqual(plainTs, entry.speech_timestamps);
      }
    });
  }

  await vad.session.release?.();
});
