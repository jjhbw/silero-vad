const assert = require('assert');
const path = require('path');
const test = require('node:test');
const { EventEmitter } = require('events');
const { Readable } = require('stream');
const childProcess = require('child_process');

function loadLibWithMockSpawn(mockSpawn) {
  const libPath = path.join(__dirname, '..', 'lib');
  const indexPath = path.join(__dirname, '..', 'index.js');
  delete require.cache[require.resolve(libPath)];
  delete require.cache[require.resolve(indexPath)];
  const originalSpawn = childProcess.spawn;
  childProcess.spawn = mockSpawn;
  try {
    return require(libPath);
  } finally {
    childProcess.spawn = originalSpawn;
  }
}

function makeMockSpawn(buffers) {
  return function mockSpawn() {
    const emitter = new EventEmitter();
    const stdout = Readable.from(buffers);
    emitter.stdout = stdout;
    emitter.kill = () => {};
    stdout.on('end', () => {
      setImmediate(() => emitter.emit('close', 0));
    });
    return emitter;
  };
}

test('getSpeechTimestamps keeps segments exactly at minSpeechDurationMs', async () => {
  const sr = 16000;
  const windowSize = 512;
  const frames = 3;
  const samples = windowSize * frames;
  const audio = new Float32Array(samples);
  const buffer = Buffer.from(audio.buffer, audio.byteOffset, audio.byteLength);
  const mockSpawn = makeMockSpawn([buffer]);
  const { getSpeechTimestamps } = loadLibWithMockSpawn(mockSpawn);

  const probs = [0.9, 0.9, 0.0];
  let callIdx = 0;
  const vad = {
    sampleRate: sr,
    resetStates: () => {},
    processChunk: async () => probs[callIdx++] ?? 0.0,
  };

  const ts = await getSpeechTimestamps('input.wav', vad, {
    threshold: 0.5,
    minSpeechDurationMs: 64,
    minSilenceDurationMs: 0,
    speechPadMs: 0,
    returnSeconds: false,
    timeResolution: 3,
  });

  assert.strictEqual(ts.length, 1);
  assert.strictEqual(ts[0].start, 0);
  assert.strictEqual(ts[0].end, 1024);
});
