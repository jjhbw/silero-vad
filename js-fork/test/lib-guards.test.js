const assert = require('assert');
const test = require('node:test');

const { writeStrippedAudio } = require('..');

test('writeStrippedAudio rejects empty segments', async () => {
  await assert.rejects(
    writeStrippedAudio('input.wav', [], 16000, 'out.wav'),
    /No valid speech segments/,
  );
});

test('writeStrippedAudio rejects missing sample rate', async () => {
  const segments = [{ start: 0, end: 1.0 }];
  await assert.rejects(
    writeStrippedAudio('input.wav', segments, null, 'out.wav'),
    /Sample rate is required/,
  );
});

