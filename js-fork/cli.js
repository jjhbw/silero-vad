#!/usr/bin/env node

const { loadSileroVad, getSpeechTimestamps, decodeWithFfmpeg, WEIGHTS } = require('./lib');

(async () => {
  try {
    const args = parseArgs(process.argv.slice(2));
    if (!args.audio.length) {
      printUsage();
      process.exit(1);
    }

    const modelSpecifier = args.model || 'default';
    const vad = await loadSileroVad(modelSpecifier);
    if (!vad.defaultSampleRate) {
      throw new Error('No sample rate available for selected model. Please use a bundled model key.');
    }
    const effectiveSampleRate = vad.defaultSampleRate;

    try {
      const results = [];
      for (const audioPath of args.audio) {
        // reuse session, reset stream state per file
        vad.resetStates();
        const t0 = performance.now();
        const audio = await decodeWithFfmpeg(audioPath, { sampleRate: effectiveSampleRate });
        const t1 = performance.now();
        const timestamps = await getSpeechTimestamps(audio, vad, {
          samplingRate: effectiveSampleRate,
          threshold: args.threshold,
          returnSeconds: args.seconds,
          timeResolution: 3,
        });
        const t2 = performance.now();
        results.push({ file: audioPath, timestamps });

        const durationSeconds = audio.length / effectiveSampleRate;
        const lines = renderTimelineLines(
          timestamps,
          durationSeconds,
          args.charsPerSecond,
          120,
        );
        const secondsPerChar = 1 / args.charsPerSecond;
        console.info(
          `legend: # speech  . silence  (1 char = ${secondsPerChar.toFixed(2)}s)  duration ${formatDuration(durationSeconds)}`,
        );
        for (const line of lines) {
          console.info(line);
        }

        const mem = process.memoryUsage();
        const toMB = (b) => (b / (1024 * 1024)).toFixed(2);
        console.info(
          [
            `file=${audioPath}`,
            `decode_ms=${(t1 - t0).toFixed(2)}`,
            `vad_ms=${(t2 - t1).toFixed(2)}`,
            `rss_mb=${toMB(mem.rss)}`,
            `heapUsed_mb=${toMB(mem.heapUsed)}`,
            `external_mb=${toMB(mem.external)}`,
          ].join(' '),
        );
      }
    } finally {
      // Keep cleanup explicit so the pattern is clear for long-lived processes.
      await vad.session.release?.();
    }
  } catch (err) {
    console.error(err.message || err);
    process.exit(1);
  }
})();

function parseArgs(argv) {
  const out = {
    model: null,
    audio: [],
    threshold: 0.5,
    seconds: true,
    charsPerSecond: 4,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--model') {
      out.model = argv[i + 1];
      i += 1;
    } else if (arg === '--audio') {
      out.audio.push(argv[i + 1]);
      i += 1;
    } else if (arg === '--threshold') {
      out.threshold = parseFloat(argv[i + 1]);
      i += 1;
    } else if (arg === '--seconds') {
      out.seconds = true;
    } else if (arg === '--cps') {
      const value = parseFloat(argv[i + 1]);
      if (Number.isFinite(value) && value > 0) {
        out.charsPerSecond = value;
      }
      i += 1;
    } else if (arg === '--help' || arg === '-h') {
      printUsage();
      process.exit(0);
    }
  }

  return out;
}

function printUsage() {
  console.log(`Usage: silero-vad-cli --audio path/to/audio [options]

Options:
  --model <key|path>    Model key (${Object.keys(WEIGHTS).join(', ')}) or custom path (default: default)
  --threshold <float>    Speech probability threshold (default: 0.5)
  --seconds              Output timestamps in seconds (default: on)
  --cps <float>          Timeline chars per second (default: 4)
  -h, --help             Show this message`);
}

function renderTimelineLines(timestamps, durationSeconds, charsPerSecond, maxLineWidth) {
  if (!durationSeconds || durationSeconds <= 0 || charsPerSecond <= 0) {
    return ['[no audio]'];
  }

  const width = Math.max(1, Math.ceil(durationSeconds * charsPerSecond));
  const slots = new Array(width).fill('.');
  for (const { start, end } of timestamps) {
    const startIdx = Math.max(0, Math.floor((start / durationSeconds) * width));
    const endIdx = Math.min(width, Math.ceil((end / durationSeconds) * width));
    for (let i = startIdx; i < endIdx; i += 1) {
      slots[i] = '#';
    }
  }

  if (!maxLineWidth || maxLineWidth <= 0) {
    return [`|${slots.join('')}|`];
  }

  const lines = [];
  for (let i = 0; i < slots.length; i += maxLineWidth) {
    lines.push(`|${slots.slice(i, i + maxLineWidth).join('')}|`);
  }
  return lines;
}

function formatDuration(seconds) {
  const whole = Math.max(0, Math.round(seconds));
  const mins = Math.floor(whole / 60);
  const secs = String(whole % 60).padStart(2, '0');
  return `${mins}:${secs}`;
}
