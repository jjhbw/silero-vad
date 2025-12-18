#!/usr/bin/env node

const { loadSileroVad, getSpeechTimestamps, decodeWithFfmpeg, WEIGHTS } = require('./lib');

// Simple CLI: node cli.js --model path/to/model.onnx --audio path/to/file [--threshold 0.5] [--seconds]
(async () => {
  try {
    const args = parseArgs(process.argv.slice(2));
    if (!args.audio.length) {
      printUsage();
      process.exit(1);
    }

    const modelSpecifier = args.model || 'default';
    const vad = await loadSileroVad(modelSpecifier);

    try {
      const results = [];
      for (const audioPath of args.audio) {
        // reuse session, reset stream state per file
        vad.resetStates();
        const audio = await decodeWithFfmpeg(audioPath, { sampleRate: args.sampleRate });
        const timestamps = await getSpeechTimestamps(audio, vad, {
          samplingRate: args.sampleRate,
          threshold: args.threshold,
          returnSeconds: args.seconds,
          timeResolution: 3,
        });
        results.push({ file: audioPath, timestamps });
      }
      console.log(JSON.stringify(results, null, 2));
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
    sampleRate: 16000,
    seconds: true,
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
    } else if (arg === '--sampleRate') {
      out.sampleRate = parseInt(argv[i + 1], 10);
      i += 1;
    } else if (arg === '--seconds') {
      out.seconds = true;
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
  --sampleRate <int>     Target sample rate for decoding (default: 16000)
  --seconds              Output timestamps in seconds (default: on)
  -h, --help             Show this message`);
}
