#!/usr/bin/env node

const fs = require('fs');
const fsp = fs.promises;
const os = require('os');
const path = require('path');
const { performance } = require('perf_hooks');
const {
  loadSileroVad,
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
}) {
  const e2eStart = performance.now();
  console.info(`file=${audioPath}`);
  console.info(`model_sample_rate=${sampleRate}`);
  if (warmup > 0) {
    await runWarmup({ audioPath, vad, sampleRate, warmup, vadOptions });
  }

  const memStats = createMemStats();
  recordMemoryUsage(memStats);

  const vadTimes = [];
  for (let i = 0; i < runs; i += 1) {
    const t0 = performance.now();
    await getSpeechTimestampsFromFfmpeg(audioPath, vad, vadOptions);
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
    const t1 = performance.now();
    stripTimes.push(t1 - t0);
    if (outputPath) {
      await fsp.unlink(outputPath);
    }
    recordMemoryUsage(memStats);
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

async function runWarmup({ audioPath, vad, sampleRate, warmup, vadOptions }) {
  for (let i = 0; i < warmup; i += 1) {
    await getSpeechTimestampsFromFfmpeg(audioPath, vad, vadOptions);
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
