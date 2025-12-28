const assert = require('assert');
const test = require('node:test');

const { getSpeechDurationSeconds, parseArgs } = require('../cli');

test('getSpeechDurationSeconds returns 0 for empty timestamps', () => {
  assert.strictEqual(getSpeechDurationSeconds([], true, 16000), 0);
  assert.strictEqual(getSpeechDurationSeconds([], false, 16000), 0);
});

test('getSpeechDurationSeconds sums seconds correctly', () => {
  const timestamps = [
    { start: 0, end: 1.5 },
    { start: 2.0, end: 3.25 },
  ];
  assert.strictEqual(getSpeechDurationSeconds(timestamps, true, 16000), 2.75);
});

test('getSpeechDurationSeconds requires sampleRate for sample timestamps', () => {
  const timestamps = [{ start: 0, end: 16000 }];
  assert.throws(
    () => getSpeechDurationSeconds(timestamps, false, null),
    /Need sampleRate/,
  );
});

test('parseArgs rejects missing --audio value', () => {
  assert.throws(
    () => parseArgs(['--audio']),
    /Missing value for --audio/,
  );
});

test('parseArgs rejects invalid --threshold value', () => {
  assert.throws(
    () => parseArgs(['--audio', 'file.wav', '--threshold', 'nope']),
    /Invalid value for --threshold/,
  );
});

