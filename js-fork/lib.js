const path = require('path');
const { spawn } = require('child_process');
const ort = require('onnxruntime-node');

/**
 * Bundled model spec map keyed by CLI/library names.
 * @type {Record<string, {path: string, sampleRate: number}>}
 */
const WEIGHTS = {
  default: { path: path.join(__dirname, 'weights', 'silero_vad_16k_op15.onnx'), sampleRate: 16000 },
  '16k': { path: path.join(__dirname, 'weights', 'silero_vad_16k_op15.onnx'), sampleRate: 16000 },
  '8k_16k': { path: path.join(__dirname, 'weights', 'silero_vad.onnx'), sampleRate: 16000 }, // decode to 16k by default
  half: { path: path.join(__dirname, 'weights', 'silero_vad_half.onnx'), sampleRate: 16000 },
  op18: { path: path.join(__dirname, 'weights', 'silero_vad_op18_ifless.onnx'), sampleRate: 16000 },
};

// Bench sweep on long-form audio showed this CPU-only config is the best general default.
const DEFAULT_SESSION_OPTIONS = {
  intraOpNumThreads: 4,
  interOpNumThreads: 1,
  executionMode: 'sequential',
  graphOptimizationLevel: 'all',
  enableCpuMemArena: true,
  enableMemPattern: true,
};

// Minimal get_speech_timestamps port that runs the Silero VAD ONNX model in Node.
class SileroVad {
  constructor(session) {
    this.session = session;
    this.outputNames = session.outputNames;
    this.resetStates();
  }

  resetStates() {
    this.state = new Float32Array(2 * 1 * 128); // shape: [2, 1, 128]
    this.context = null;
    this.lastSr = null;
    this.contextSize = null;
    this.inputWithContext = null;
    this.srTensor = null;
  }

  async processChunk(chunk, sampleRate) {
    const sr = sampleRate;
    const windowSize = sr === 16000 ? 512 : 256;
    const contextSize = sr === 16000 ? 64 : 32;

    if (chunk.length !== windowSize) {
      throw new Error(`Expected chunk of ${windowSize} samples, got ${chunk.length}`);
    }

    // Reset state when sample rate changes.
    if (this.lastSr && this.lastSr !== sr) {
      this.resetStates();
    }

    if (!this.context || this.contextSize !== contextSize) {
      this.contextSize = contextSize;
      this.context = new Float32Array(contextSize); // zeros
      this.inputWithContext = new Float32Array(contextSize + windowSize);
      this.srTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(sr)]));
    }

    const inputWithContext = this.inputWithContext;
    inputWithContext.set(this.context, 0);
    inputWithContext.set(chunk, contextSize);

    const feeds = {
      input: new ort.Tensor('float32', inputWithContext, [1, inputWithContext.length]),
      state: new ort.Tensor('float32', this.state, [2, 1, 128]),
      sr: this.srTensor,
    };

    const results = await this.session.run(feeds);
    const probTensor = results[this.outputNames[0]];
    const newStateTensor = results[this.outputNames[1]];

    this.state.set(newStateTensor.data);
    this.context.set(inputWithContext.subarray(inputWithContext.length - contextSize));
    this.lastSr = sr;

    return probTensor.data[0];
  }
}

/**
 * Load a Silero VAD ONNX model and return a ready-to-run VAD instance.
 * @param {string} [model='default'] Bundled model key or custom ONNX path.
 * @param {Object} [opts]
 * @param {Object} [opts.sessionOptions] onnxruntime-node session options override.
 * @returns {Promise<SileroVad>}
 */
async function loadSileroVad(model = 'default', opts = {}) {
  const spec = WEIGHTS[model];
  const modelPath = spec ? spec.path : model || WEIGHTS.default.path;
  const sessionOptions = {
    ...DEFAULT_SESSION_OPTIONS,
    ...(opts.sessionOptions || {}),
  };
  const session = await ort.InferenceSession.create(modelPath, sessionOptions);
  const vad = new SileroVad(session);
  vad.sampleRate = spec ? spec.sampleRate : null;
  return vad;
}

/**
 * Run VAD on an audio file and return speech segments.
 * @param {string} inputPath
 * @param {SileroVad} vad
 * @param {Object} [options]
 * @param {number} [options.threshold=0.5] Start speech when prob >= threshold.
 *   Example: if probs hover at 0.45-0.6, threshold=0.6 will miss soft speech.
 * @param {number} [options.minSpeechDurationMs=250] Drop segments shorter than this.
 *   Example: a 120 ms burst above threshold is discarded at 250 ms.
 * @param {number} [options.minSilenceDurationMs=100] End speech only after silence
 *   stays below negThreshold for this long.
 *   Example: a 50 ms pause will not split a segment at 100 ms.
 * @param {number} [options.speechPadMs=30] Pad each segment on both sides, clamped
 *   to neighbors. Example: [1.000, 2.000] -> ~[0.970, 2.030].
 * @param {number} [options.timeResolution=3] Decimal places for seconds output.
 *   Example: timeResolution=1 turns 1.23456 into 1.2.
 * @param {number} [options.negThreshold=threshold-0.15] End speech when prob dips
 *   below this; provides hysteresis vs threshold. Example: threshold=0.5,
 *   negThreshold=0.35 keeps speech open during brief 0.4 dips.
 *   Default clamps to >= 0.01 to avoid an always-on end condition.
 * @param {number} [options.sampleRate]
 * @returns {Promise<Array<{start: number, end: number, startSample: number, endSample: number}>>}
 *   start/end are seconds; startSample/endSample are sample indices.
 */
async function getSpeechTimestamps(
  inputPath,
  vad,
  {
    threshold = 0.5,
    minSpeechDurationMs = 250,
    minSilenceDurationMs = 100,
    speechPadMs = 30,
    timeResolution = 3,
    negThreshold,
    sampleRate,
  } = {},
) {
  if (!vad) {
    throw new Error('Pass a loaded SileroVad instance');
  }

  const sr = sampleRate || vad.sampleRate;
  if (!sr) {
    throw new Error('VAD sample rate is undefined. Use a bundled model key.');
  }

  if (sr !== 8000 && sr !== 16000) {
    throw new Error('Supported sampling rates: 8000 or 16000 (or a multiple of 16000).');
  }

  const windowSize = sr === 16000 ? 512 : 256;
  const minSpeechSamples = (sr * minSpeechDurationMs) / 1000;
  const minSilenceSamples = (sr * minSilenceDurationMs) / 1000;
  const speechPadSamples = (sr * speechPadMs) / 1000;
  const negThres = negThreshold ?? Math.max(threshold - 0.15, 0.01);

  vad.resetStates();

  let triggered = false;
  let tempEnd = 0;
  let currentSpeech = {};
  const speeches = [];
  let processedSamples = 0;
  let totalSamples = 0;
  let leftoverBytes = Buffer.alloc(0);
  const frameScratch = new Float32Array(windowSize);
  let pendingLen = 0;

  const channels = 1;
  const args = [
    '-v',
    'error',
    '-i',
    inputPath,
    '-ac',
    String(channels),
    '-ar',
    String(sr),
    '-f',
    'f32le',
    'pipe:1',
  ];
  const ffmpeg = spawn('ffmpeg', args, { stdio: ['ignore', 'pipe', 'inherit'] });

  const processFrame = async (frame, curSample) => {
    const speechProb = await vad.processChunk(frame, sr);
    if (speechProb >= threshold && tempEnd) {
      tempEnd = 0;
    }

    if (speechProb >= threshold && !triggered) {
      triggered = true;
      currentSpeech.start = curSample;
      return;
    }

    if (speechProb < negThres && triggered) {
      if (!tempEnd) {
        tempEnd = curSample;
      }
      if (curSample - tempEnd < minSilenceSamples) {
        return;
      }

      currentSpeech.end = tempEnd;
      if (currentSpeech.end - currentSpeech.start >= minSpeechSamples) {
        speeches.push(currentSpeech);
      }
      currentSpeech = {};
      triggered = false;
      tempEnd = 0;
    }
  };

  const streamDone = (async () => {
    for await (const chunk of ffmpeg.stdout) {
      let data = chunk;
      if (leftoverBytes.length) {
        const combined = Buffer.alloc(leftoverBytes.length + chunk.length);
        leftoverBytes.copy(combined, 0);
        chunk.copy(combined, leftoverBytes.length);
        data = combined;
        leftoverBytes = Buffer.alloc(0);
      }

      const usableBytes = data.length - (data.length % 4);
      if (usableBytes <= 0) {
        leftoverBytes = data;
        continue;
      }

      leftoverBytes = data.subarray(usableBytes);
      const floatData = new Float32Array(
        data.buffer,
        data.byteOffset,
        usableBytes / Float32Array.BYTES_PER_ELEMENT,
      );
      totalSamples += floatData.length;

      let offset = 0;
      if (pendingLen) {
        const needed = windowSize - pendingLen;
        if (floatData.length >= needed) {
          frameScratch.set(frameScratch.subarray(0, pendingLen), 0);
          frameScratch.set(floatData.subarray(0, needed), pendingLen);
          const curSample = processedSamples;
          processedSamples += windowSize;
          await processFrame(frameScratch, curSample);
          offset = needed;
          pendingLen = 0;
        } else {
          frameScratch.set(floatData, pendingLen);
          pendingLen += floatData.length;
          continue;
        }
      }

      while (offset + windowSize <= floatData.length) {
        const frame = floatData.subarray(offset, offset + windowSize);
        const curSample = processedSamples;
        processedSamples += windowSize;
        await processFrame(frame, curSample);
        offset += windowSize;
      }

      const remainingSamples = floatData.length - offset;
      if (remainingSamples > 0) {
        frameScratch.set(floatData.subarray(offset), 0);
        pendingLen = remainingSamples;
      } else {
        pendingLen = 0;
      }
    }
  })();

  await new Promise((resolve, reject) => {
    let settled = false;
    const finish = (fn) => (value) => {
      if (settled) {
        return;
      }
      settled = true;
      fn(value);
    };
    const resolveOnce = finish(resolve);
    const rejectOnce = finish(reject);

    streamDone.then(resolveOnce, (err) => {
      ffmpeg.kill('SIGKILL');
      rejectOnce(err);
    });

    ffmpeg.on('error', rejectOnce);
    ffmpeg.on('close', (code) => {
      if (code !== 0) {
        rejectOnce(new Error(`ffmpeg exited with code ${code}`));
        return;
      }
      streamDone.then(resolveOnce, rejectOnce);
    });
  });

  if (leftoverBytes.length) {
    const usableBytes = leftoverBytes.length - (leftoverBytes.length % 4);
    if (usableBytes > 0) {
      const tailFloats = new Float32Array(
        leftoverBytes.buffer,
        leftoverBytes.byteOffset,
        usableBytes / Float32Array.BYTES_PER_ELEMENT,
      );
      if (tailFloats.length) {
        frameScratch.set(tailFloats, pendingLen);
        pendingLen += tailFloats.length;
      }
    }
  }

  if (pendingLen) {
    const padded = new Float32Array(windowSize);
    padded.set(frameScratch.subarray(0, pendingLen));
    const curSample = processedSamples;
    await processFrame(padded, curSample);
    processedSamples += windowSize;
  }

  if (currentSpeech.start !== undefined) {
    currentSpeech.end = totalSamples;
    if (currentSpeech.end - currentSpeech.start >= minSpeechSamples) {
      speeches.push(currentSpeech);
    }
  }

  for (let idx = 0; idx < speeches.length; idx += 1) {
    const speech = speeches[idx];
    const prevEnd = idx === 0 ? 0 : speeches[idx - 1].end;
    const nextStart = idx === speeches.length - 1 ? totalSamples : speeches[idx + 1].start;
    const padStart = Math.max(speech.start - speechPadSamples, prevEnd);
    const padEnd = Math.min(speech.end + speechPadSamples, nextStart);
    speech.start = Math.max(0, Math.floor(padStart));
    speech.end = Math.min(totalSamples, Math.floor(padEnd));
  }

  const convertSeconds = (samples) => +(samples / sr).toFixed(timeResolution);
  const result = speeches.map(({ start, end }) => ({
    start: convertSeconds(start),
    end: convertSeconds(end),
    startSample: start,
    endSample: end,
  }));

  return result;
}

/**
 * Write a new audio file containing only the provided speech segments.
 * Uses ffmpeg; encoding is inferred from outputPath extension/container.
 * @param {string} inputPath
 * @param {Array<{start: number, end: number}>} segmentsSeconds Seconds-based ranges.
 * @param {number} sampleRate Output sample rate (required by ffmpeg).
 * @param {string} outputPath
 * @returns {Promise<void>}
 */
async function writeStrippedAudio(inputPath, segmentsSeconds, sampleRate, outputPath) {
  if (!segmentsSeconds || !segmentsSeconds.length) {
    throw new Error('No valid speech segments to write');
  }
  if (!sampleRate) {
    throw new Error('Sample rate is required to write WAV');
  }
  const expr = segmentsSeconds
    .map(({ start, end }) => `between(t\\,${start.toFixed(6)}\\,${end.toFixed(6)})`)
    .join('+');
  const filter = `aselect='${expr}',asetpts=N/SR/TB`;

  const args = [
    '-y',
    '-v',
    'error',
    '-i',
    inputPath,
    '-af',
    filter,
    '-ac',
    '1',
    '-ar',
    String(sampleRate),
    outputPath,
  ];

  await new Promise((resolve, reject) => {
    const ffmpeg = spawn('ffmpeg', args, { stdio: ['ignore', 'ignore', 'inherit'] });
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

module.exports = {
  loadSileroVad,
  getSpeechTimestamps,
  writeStrippedAudio,
  WEIGHTS,
};
