#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const { loadSileroVad, getSpeechTimestamps, decodeWithFfmpeg, WEIGHTS } = require('./lib');

const toMB = (b) => (b / (1024 * 1024)).toFixed(2);

(async () => {
  try {
    const args = parseArgs(process.argv.slice(2));
    if (!args.audio.length) {
      printUsage();
      process.exit(1);
    }

    const modelSpecifier = args.model || 'default';
    const vad = await loadSileroVad(modelSpecifier);
    if (!vad.sampleRate) {
      throw new Error('No sample rate available for selected model. Please use a bundled model key.');
    }
    const effectiveSampleRate = vad.sampleRate;

    try {
      const results = [];
      for (const audioPath of args.audio) {
        // reuse session, reset stream state per file
        vad.resetStates();
        const t0 = performance.now();
        const audio = await decodeWithFfmpeg(audioPath, { sampleRate: effectiveSampleRate });
        const t1 = performance.now();
        const timestamps = await getSpeechTimestamps(audio, vad, {
          threshold: args.threshold,
          minSpeechDurationMs: args.minSpeechDurationMs,
          minSilenceDurationMs: args.minSilenceDurationMs,
          speechPadMs: args.speechPadMs,
          returnSeconds: args.seconds,
          timeResolution: args.timeResolution,
          negThreshold: args.negThreshold,
        });
        const t2 = performance.now();
        results.push({ file: audioPath, timestamps });

        const mem = process.memoryUsage();
        const speechSeconds = getSpeechDurationSeconds(
          timestamps,
          args.seconds,
          effectiveSampleRate,
        );
        const durationSeconds = audio.length / effectiveSampleRate;
        const silenceSeconds = Math.max(0, durationSeconds - speechSeconds);
        const totalForPct = durationSeconds > 0 ? durationSeconds : 1;
        const speechPct = (speechSeconds / totalForPct) * 100;
        const silencePct = (silenceSeconds / totalForPct) * 100;
        const lines = renderTimelineLines(
          timestamps,
          durationSeconds,
          args.charsPerSecond,
          120,
        );
        const secondsPerChar = 1 / args.charsPerSecond;
        console.info(
          [
            `file=${audioPath}`,
            `duration=${formatDuration(durationSeconds)}`,
          ].join(' '),
        );
        console.info([
          `speech=${speechSeconds.toFixed(2)}s (${speechPct.toFixed(1)}%)`,
          `silence=${silenceSeconds.toFixed(2)}s (${silencePct.toFixed(1)}%)`, ,
          `total=${durationSeconds.toFixed(2)}s`
        ].join(' ')
        );
        const totalMs = t2 - t0;
        const totalForPctMs = totalMs > 0 ? totalMs : 1;
        const decodePct = ((t1 - t0) / totalForPctMs) * 100;
        const vadPct = ((t2 - t1) / totalForPctMs) * 100;
        console.info(
          [
            `decode_took=${(t1 - t0).toFixed(2)} (${decodePct.toFixed(1)}%)`,
            `vad_took=${(t2 - t1).toFixed(2)} (${vadPct.toFixed(1)}%)`,
          ].join(' '),
        );
        console.info(
          [
            `rss_mb=${toMB(mem.rss)}`,
            `heapUsed_mb=${toMB(mem.heapUsed)}`,
            `external_mb=${toMB(mem.external)}`,
          ].join(' '),
        );
        console.info(
          `legend: # speech  . silence  (1 char = ${secondsPerChar.toFixed(2)}s)`,
        );
        for (const line of lines) {
          console.info(line);
        }

        if (args.stripSilence) {
          const segmentsSeconds = timestamps.map(({ start, end, startSeconds, endSeconds }) => ({
            start: args.seconds ? start : startSeconds,
            end: args.seconds ? end : endSeconds,
          }));
          if (!segmentsSeconds.length) {
            console.info(`strip_silence=skipped (no speech detected)`);
          } else {
            if (args.outputDir && !fs.existsSync(args.outputDir)) {
              fs.mkdirSync(args.outputDir, { recursive: true });
            }
            const outputPath = ensureUniquePath(
              getStripOutputPath(audioPath, args.outputDir),
            );
            const stripT0 = performance.now();
            await writeStrippedAudio(audioPath, segmentsSeconds, outputPath);
            const stripT1 = performance.now();
            const strippedSeconds = segmentsSeconds.reduce(
              (sum, seg) => sum + (seg.end - seg.start),
              0,
            );
            console.info(
              `strip_silence_output=${outputPath} duration=${strippedSeconds.toFixed(2)}s`,
            );
            console.info(`strip_silence_took=${(stripT1 - stripT0).toFixed(2)}ms`);
          }
        }
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
    minSpeechDurationMs: 250,
    minSilenceDurationMs: 100,
    speechPadMs: 30,
    timeResolution: 3,
    negThreshold: null,
    seconds: true,
    charsPerSecond: 4,
    stripSilence: false,
    outputDir: null,
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
    } else if (arg === '--min-speech-ms') {
      const value = parseFloat(argv[i + 1]);
      if (Number.isFinite(value) && value >= 0) {
        out.minSpeechDurationMs = value;
      }
      i += 1;
    } else if (arg === '--min-silence-ms') {
      const value = parseFloat(argv[i + 1]);
      if (Number.isFinite(value) && value >= 0) {
        out.minSilenceDurationMs = value;
      }
      i += 1;
    } else if (arg === '--speech-pad-ms') {
      const value = parseFloat(argv[i + 1]);
      if (Number.isFinite(value) && value >= 0) {
        out.speechPadMs = value;
      }
      i += 1;
    } else if (arg === '--time-resolution') {
      const value = parseInt(argv[i + 1], 10);
      if (Number.isFinite(value) && value >= 0) {
        out.timeResolution = value;
      }
      i += 1;
    } else if (arg === '--neg-threshold') {
      const value = parseFloat(argv[i + 1]);
      if (Number.isFinite(value)) {
        out.negThreshold = value;
      }
      i += 1;
    } else if (arg === '--seconds') {
      out.seconds = true;
    } else if (arg === '--cps') {
      const value = parseFloat(argv[i + 1]);
      if (Number.isFinite(value) && value > 0) {
        out.charsPerSecond = value;
      }
      i += 1;
    } else if (arg === '--strip-silence') {
      out.stripSilence = true;
    } else if (arg === '--output-dir') {
      out.outputDir = argv[i + 1];
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
  --min-speech-ms <ms>   Minimum speech duration in ms (default: 250)
  --min-silence-ms <ms>  Minimum silence duration in ms (default: 100)
  --speech-pad-ms <ms>   Padding added to speech segments in ms (default: 30)
  --time-resolution <n>  Decimal places for seconds output (default: 3)
  --neg-threshold <f>    Negative threshold override (default: threshold - 0.15)
  --seconds              Output timestamps in seconds (default: on)
  --cps <float>          Timeline chars per second (default: 4)
  --strip-silence         Write a new file with all silences removed
  --output-dir <path>     Output directory for strip-silence files (default: input dir)
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

function getSpeechDurationSeconds(timestamps, timestampsInSeconds, sampleRate) {
  if (!timestamps || !timestamps.length) {
    new Error("Need timestamps")
  }
  if (!sampleRate) {
    new Error("Need sampleRate")
  }
  if (timestampsInSeconds) {
    return timestamps.reduce((sum, { start, end }) => sum + (end - start), 0);
  }
  return timestamps.reduce((sum, { start, end }) => sum + (end - start) / sampleRate, 0);
}

function formatDuration(seconds) {
  const whole = Math.max(0, Math.round(seconds));
  const mins = Math.floor(whole / 60);
  const secs = String(whole % 60).padStart(2, '0');
  return `${mins}:${secs}`;
}

function getStripOutputPath(inputPath, outputDir) {
  const dir = outputDir || path.dirname(inputPath);
  const ext = path.extname(inputPath);
  const base = path.basename(inputPath, ext);
  const safeExt = ext || '.wav';
  return path.join(dir, `${base}_speech${safeExt}`);
}

function ensureUniquePath(outputPath) {
  if (!fs.existsSync(outputPath)) {
    return outputPath;
  }
  const dir = path.dirname(outputPath);
  const ext = path.extname(outputPath);
  const base = path.basename(outputPath, ext);
  for (let i = 1; ; i += 1) {
    const candidate = path.join(dir, `${base}-${i}${ext}`);
    if (!fs.existsSync(candidate)) {
      return candidate;
    }
  }
}

function buildConcatFilter(segmentsSeconds) {
  const parts = [];
  const labels = [];
  let idx = 0;
  for (const { start, end } of segmentsSeconds) {
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
      continue;
    }
    const label = `a${idx}`;
    parts.push(
      `[0:a]atrim=start=${start}:end=${end},asetpts=PTS-STARTPTS[${label}]`,
    );
    labels.push(`[${label}]`);
    idx += 1;
  }
  if (!labels.length) {
    return null;
  }
  parts.push(`${labels.join('')}concat=n=${labels.length}:v=0:a=1[outa]`);
  return parts.join(';');
}

function writeStrippedAudio(inputPath, segmentsSeconds, outputPath) {
  return new Promise((resolve, reject) => {
    const filter = buildConcatFilter(segmentsSeconds);
    if (!filter) {
      reject(new Error('No valid speech segments to write'));
      return;
    }
    const args = [
      '-v',
      'error',
      '-i',
      inputPath,
      '-filter_complex',
      filter,
      '-map',
      '[outa]',
      outputPath,
    ];
    const ffmpeg = spawn('ffmpeg', args, { stdio: ['ignore', 'inherit', 'inherit'] });
    ffmpeg.on('error', reject);
    ffmpeg.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`ffmpeg exited with code ${code}`));
        return;
      }
      resolve();
    });
  });
}
