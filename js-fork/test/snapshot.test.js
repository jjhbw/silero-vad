const assert = require('assert');
const fs = require('fs');
const path = require('path');
const test = require('node:test');

const { execFile } = require('child_process');

const ROOT = path.join(__dirname, '..', '..');
const SNAPSHOT_DIR = path.join(ROOT, 'tests', 'snapshots');
const DATA_DIR = path.join(ROOT, 'tests', 'data');
const CLI = path.join(__dirname, '..', 'cli.js');

function readSnapshot(name) {
  const p = path.join(SNAPSHOT_DIR, name);
  const raw = fs.readFileSync(p, 'utf8');
  return JSON.parse(raw);
}

function runCli(audioPath, sampleRate) {
  return new Promise((resolve, reject) => {
    const args = ['--audio', audioPath, '--sampleRate', String(sampleRate)];
    const child = execFile('node', [CLI, ...args], { encoding: 'utf8' }, (err, stdout, stderr) => {
      if (err) {
        err.stderr = stderr;
        reject(err);
        return;
      }
      try {
        const parsed = JSON.parse(stdout);
        resolve(parsed);
      } catch (parseErr) {
        reject(parseErr);
      }
    });
  });
}

test('onnx snapshot matches python ground truth', async () => {
  const snapshot = readSnapshot('onnx.json');

  for (const entry of snapshot.snapshots) {
    await test(`file ${entry.file}`, async () => {
      const wavPath = path.join(DATA_DIR, entry.file);
      const ts = await runCli(wavPath, entry.sampling_rate);
      if (entry.file === 'test.mp3') {
        // MP3 decoding in Node (ffmpeg) vs Python (torchaudio) can differ by encoder delay/padding,
        // so allow a small tolerance instead of strict equality.
        assert.strictEqual(ts.length, entry.speech_timestamps.length);
        for (let i = 0; i < ts.length; i += 1) {
          const a = ts[i];
          const e = entry.speech_timestamps[i];
          assert.ok(Math.abs(a.start - e.start) <= 0.1, `start mismatch for ${entry.file}`);
          assert.ok(Math.abs(a.end - e.end) <= 0.1, `end mismatch for ${entry.file}`);
        }
      } else {
        assert.deepStrictEqual(ts, entry.speech_timestamps);
      }
    });
  }
});
