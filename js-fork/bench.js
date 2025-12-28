#!/usr/bin/env node

const fs = require('fs');
const fsp = fs.promises;
const os = require('os');
const path = require('path');
const { performance } = require('perf_hooks');
const {
  loadSileroVad,
  decodeWithFfmpeg,
  getSpeechTimestamps,
  getSpeechTimestampsFromFfmpeg,
  writeStrippedAudioWithFfmpeg,
  WEIGHTS,
} = require('./lib');

(async () => {
  try {
    const args = parseArgs(process.argv.slice(2));
    if (!args.audio.length) {
      printUsage();
      process.exit(1);
    }

    const modelSpecifier = args.model || 'default';
    const sessionOptions = buildSessionOptions(args);
    const vad = await loadSileroVad(
      modelSpecifier,
      sessionOptions ? { sessionOptions } : undefined,
    );
    if (!vad.sampleRate) {
      throw new Error('No sample rate available for selected model. Please use a bundled model key.');
    }
    const effectiveSampleRate = vad.sampleRate;

    const outputDir = await ensureOutputDir();

    try {
      for (const audioPath of args.audio) {
        await runBenchmarks({
          audioPath,
          vad,
          sampleRate: effectiveSampleRate,
          outputDir,
          runs: args.runs,
          warmup: args.warmup,
          vadOptions: {
            threshold: args.threshold,
            minSpeechDurationMs: args.minSpeechDurationMs,
            minSilenceDurationMs: args.minSilenceDurationMs,
            speechPadMs: args.speechPadMs,
            returnSeconds: true,
            timeResolution: args.timeResolution,
            negThreshold: args.negThreshold,
          },
          streaming: args.streaming,
        });
      }
    } finally {
      await vad.session.release?.();
      await fsp.rm(outputDir, { recursive: true, force: true });
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
    runs: 5,
    warmup: 1,
    threshold: 0.5,
    minSpeechDurationMs: 250,
    minSilenceDurationMs: 100,
    speechPadMs: 30,
    timeResolution: 3,
    negThreshold: null,
    streaming: false,
    intraOpNumThreads: null,
    interOpNumThreads: null,
    executionMode: null,
    graphOptimizationLevel: null,
    enableCpuMemArena: null,
    enableMemPattern: null,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--model') {
      out.model = argv[i + 1];
      i += 1;
    } else if (arg === '--audio') {
      out.audio.push(argv[i + 1]);
      i += 1;
    } else if (arg === '--runs') {
      const value = parseInt(argv[i + 1], 10);
      if (Number.isFinite(value) && value > 0) {
        out.runs = value;
      }
      i += 1;
    } else if (arg === '--warmup') {
      const value = parseInt(argv[i + 1], 10);
      if (Number.isFinite(value) && value >= 0) {
        out.warmup = value;
      }
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
    } else if (arg === '--streaming') {
      out.streaming = true;
    } else if (arg === '--intra-threads') {
      const value = parseInt(argv[i + 1], 10);
      if (Number.isFinite(value)) {
        out.intraOpNumThreads = value;
      }
      i += 1;
    } else if (arg === '--inter-threads') {
      const value = parseInt(argv[i + 1], 10);
      if (Number.isFinite(value)) {
        out.interOpNumThreads = value;
      }
      i += 1;
    } else if (arg === '--execution-mode') {
      out.executionMode = argv[i + 1];
      i += 1;
    } else if (arg === '--graph-opt') {
      out.graphOptimizationLevel = argv[i + 1];
      i += 1;
    } else if (arg === '--enable-cpu-mem-arena') {
      out.enableCpuMemArena = parseBool(argv[i + 1]);
      i += 1;
    } else if (arg === '--enable-mem-pattern') {
      out.enableMemPattern = parseBool(argv[i + 1]);
      i += 1;
    } else if (arg === '--help' || arg === '-h') {
      printUsage();
      process.exit(0);
    }
  }

  return out;
}

function printUsage() {
  console.log(`Usage: silero-vad-bench --audio path/to/audio [options]

Options:
  --model <key|path>      Model key (${Object.keys(WEIGHTS).join(', ')}) or custom path (default: default)
  --runs <n>              Number of timed runs per benchmark (default: 5)
  --warmup <n>            Warmup runs before timing (default: 1)
  --threshold <float>     Speech probability threshold (default: 0.5)
  --min-speech-ms <ms>    Minimum speech duration in ms (default: 250)
  --min-silence-ms <ms>   Minimum silence duration in ms (default: 100)
  --speech-pad-ms <ms>    Padding added to speech segments in ms (default: 30)
  --time-resolution <n>   Decimal places for seconds output (default: 3)
  --neg-threshold <f>     Negative threshold override (default: threshold - 0.15)
  --streaming             Use ffmpeg streaming decode for VAD/strip benchmarks
  --intra-threads <n>     ORT intra-op thread count
  --inter-threads <n>     ORT inter-op thread count
  --execution-mode <m>    ORT execution mode: sequential | parallel
  --graph-opt <level>     ORT graph optimization: disabled | basic | extended | layout | all
  --enable-cpu-mem-arena <bool>  ORT CPU memory arena on/off
  --enable-mem-pattern <bool>    ORT memory pattern on/off
  -h, --help              Show this message`);
}

function parseBool(value) {
  if (value === undefined) {
    return null;
  }
  const normalized = String(value).toLowerCase();
  if (normalized === 'true' || normalized === '1' || normalized === 'yes') {
    return true;
  }
  if (normalized === 'false' || normalized === '0' || normalized === 'no') {
    return false;
  }
  return null;
}

function buildSessionOptions(args) {
  const sessionOptions = {};
  if (Number.isFinite(args.intraOpNumThreads)) {
    sessionOptions.intraOpNumThreads = args.intraOpNumThreads;
  }
  if (Number.isFinite(args.interOpNumThreads)) {
    sessionOptions.interOpNumThreads = args.interOpNumThreads;
  }
  if (args.executionMode) {
    sessionOptions.executionMode = args.executionMode;
  }
  if (args.graphOptimizationLevel) {
    sessionOptions.graphOptimizationLevel = args.graphOptimizationLevel;
  }
  if (args.enableCpuMemArena !== null) {
    sessionOptions.enableCpuMemArena = args.enableCpuMemArena;
  }
  if (args.enableMemPattern !== null) {
    sessionOptions.enableMemPattern = args.enableMemPattern;
  }
  return Object.keys(sessionOptions).length ? sessionOptions : null;
}

async function runBenchmarks({
  audioPath,
  vad,
  sampleRate,
  outputDir,
  runs,
  warmup,
  vadOptions,
  streaming,
}) {
  const e2eStart = performance.now();
  console.info(`file=${audioPath}`);
  console.info(`model_sample_rate=${sampleRate}`);
  if (warmup > 0) {
    await runWarmup({ audioPath, vad, sampleRate, warmup, vadOptions, streaming });
  }

  const memStats = createMemStats();
  recordMemoryUsage(memStats);

  const decodeTimes = [];
  if (!streaming) {
    for (let i = 0; i < runs; i += 1) {
      const t0 = performance.now();
      await decodeWithFfmpeg(audioPath, { sampleRate });
      const t1 = performance.now();
      decodeTimes.push(t1 - t0);
      recordMemoryUsage(memStats);
    }
  }

  const vadTimes = [];
  for (let i = 0; i < runs; i += 1) {
    const t0 = performance.now();
    if (streaming) {
      await getSpeechTimestampsFromFfmpeg(audioPath, vad, vadOptions);
    } else {
      const audio = await decodeWithFfmpeg(audioPath, { sampleRate });
      await getSpeechTimestamps(audio, vad, vadOptions);
    }
    const t1 = performance.now();
    vadTimes.push(t1 - t0);
    recordMemoryUsage(memStats);
  }

  const stripTimes = [];
  const stripWriteTimes = [];
  let skippedStrip = 0;
  for (let i = 0; i < runs; i += 1) {
    const t0 = performance.now();
    let outputPath = null;
    if (streaming) {
      const timestamps = await getSpeechTimestampsFromFfmpeg(audioPath, vad, vadOptions);
      const segments = timestamps.map(({ start, end }) => ({ start, end }));
      if (!segments.length) {
        skippedStrip += 1;
        const t1 = performance.now();
        stripTimes.push(t1 - t0);
        stripWriteTimes.push(0);
        continue;
      }
      outputPath = path.join(
        outputDir,
        `${path.basename(audioPath, path.extname(audioPath))}_speech_${i + 1}.wav`,
      );
      const stripT0 = performance.now();
      await writeStrippedAudioWithFfmpeg(audioPath, segments, sampleRate, outputPath);
      const stripT1 = performance.now();
      stripWriteTimes.push(stripT1 - stripT0);
    } else {
      const audio = await decodeWithFfmpeg(audioPath, { sampleRate });
      const timestamps = await getSpeechTimestamps(audio, vad, vadOptions);
      const segments = timestamps.map(({ start, end }) => ({ start, end }));
      if (!segments.length) {
        skippedStrip += 1;
        const t1 = performance.now();
        stripTimes.push(t1 - t0);
        stripWriteTimes.push(0);
        continue;
      }
      outputPath = path.join(
        outputDir,
        `${path.basename(audioPath, path.extname(audioPath))}_speech_${i + 1}.wav`,
      );
      const stripT0 = performance.now();
      await writeStrippedAudio(audio, segments, sampleRate, outputPath);
      const stripT1 = performance.now();
      stripWriteTimes.push(stripT1 - stripT0);
    }
    const t1 = performance.now();
    stripTimes.push(t1 - t0);
    if (outputPath) {
      await fsp.unlink(outputPath);
    }
    recordMemoryUsage(memStats);
  }

  if (!streaming) {
    printStats('ffmpeg_decode', decodeTimes);
  } else {
    console.info('ffmpeg_decode_ms skipped (streaming mode)');
  }
  printStats('file_to_vad', vadTimes);
  printStats('file_to_stripped', stripTimes);
  printStats('strip_write', stripWriteTimes);
  if (skippedStrip) {
    console.info(`strip_skipped=${skippedStrip} (no speech detected)`);
  }
  const e2eMs = performance.now() - e2eStart;
  console.info(`end_to_end_ms total=${e2eMs.toFixed(2)}`);
  const mem = process.memoryUsage();
  console.info(
    [
      `mem_rss_mb=${toMB(mem.rss)}`,
      `mem_heapUsed_mb=${toMB(mem.heapUsed)}`,
      `mem_external_mb=${toMB(mem.external)}`,
    ].join(' '),
  );
  console.info(
    [
      `mem_rss_peak_mb=${toMB(memStats.maxRss)}`,
      `mem_heapUsed_peak_mb=${toMB(memStats.maxHeapUsed)}`,
      `mem_external_peak_mb=${toMB(memStats.maxExternal)}`,
    ].join(' '),
  );
  console.info('');
}

async function runWarmup({ audioPath, vad, sampleRate, warmup, vadOptions, streaming }) {
  for (let i = 0; i < warmup; i += 1) {
    if (streaming) {
      await getSpeechTimestampsFromFfmpeg(audioPath, vad, vadOptions);
    } else {
      const audio = await decodeWithFfmpeg(audioPath, { sampleRate });
      await getSpeechTimestamps(audio, vad, vadOptions);
    }
  }
}

function printStats(label, values) {
  const stats = calcStats(values);
  console.info(
    `${label}_ms avg=${stats.mean.toFixed(2)} min=${stats.min.toFixed(2)} max=${stats.max.toFixed(2)}`,
  );
}

function calcStats(values) {
  const safe = values.length ? values : [0];
  let sum = 0;
  let min = safe[0];
  let max = safe[0];
  for (const value of safe) {
    sum += value;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  return {
    mean: sum / safe.length,
    min,
    max,
  };
}

function toMB(bytes) {
  return (bytes / (1024 * 1024)).toFixed(2);
}

function createMemStats() {
  return {
    maxRss: 0,
    maxHeapUsed: 0,
    maxExternal: 0,
  };
}

function recordMemoryUsage(stats) {
  const mem = process.memoryUsage();
  stats.maxRss = Math.max(stats.maxRss, mem.rss);
  stats.maxHeapUsed = Math.max(stats.maxHeapUsed, mem.heapUsed);
  stats.maxExternal = Math.max(stats.maxExternal, mem.external);
  return mem;
}

async function ensureOutputDir() {
  return fsp.mkdtemp(path.join(os.tmpdir(), 'silero-vad-bench-'));
}

async function writeStrippedAudio(audio, segmentsSeconds, sampleRate, outputPath) {
  if (!audio || !audio.length) {
    throw new Error('No audio samples available to write');
  }
  if (!sampleRate) {
    throw new Error('Sample rate is required to write WAV');
  }
  const ranges = segmentsSeconds
    .map(({ start, end }) => ({
      start: Math.max(0, Math.floor(start * sampleRate)),
      end: Math.min(audio.length, Math.floor(end * sampleRate)),
    }))
    .filter(({ start, end }) => end > start);
  if (!ranges.length) {
    throw new Error('No valid speech segments to write');
  }
  let totalSamples = 0;
  for (const { start, end } of ranges) {
    totalSamples += end - start;
  }
  const out = new Float32Array(totalSamples);
  let offset = 0;
  for (const { start, end } of ranges) {
    out.set(audio.subarray(start, end), offset);
    offset += end - start;
  }
  await writeWavFile(outputPath, out, sampleRate);
}

async function writeWavFile(outputPath, samples, sampleRate) {
  const numChannels = 1;
  const bitsPerSample = 16;
  const blockAlign = (numChannels * bitsPerSample) / 8;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * 2;
  const buffer = Buffer.alloc(44 + dataSize);

  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write('WAVE', 8);
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(numChannels, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(byteRate, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(bitsPerSample, 34);
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataSize, 40);

  let writeOffset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    const int16 = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    buffer.writeInt16LE(Math.round(int16), writeOffset);
    writeOffset += 2;
  }

  await fsp.writeFile(outputPath, buffer);
}
