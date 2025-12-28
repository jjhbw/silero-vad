const fs = require('fs');
const fsp = fs.promises;
const path = require('path');
const { spawn } = require('child_process');
const ort = require('onnxruntime-node');

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
    this.resetStates();
  }

  resetStates() {
    this.state = new Float32Array(2 * 1 * 128); // shape: [2, 1, 128]
    this.context = null;
    this.lastSr = null;
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

    if (!this.context) {
      this.context = new Float32Array(contextSize); // zeros
    }

    const inputWithContext = new Float32Array(contextSize + windowSize);
    inputWithContext.set(this.context, 0);
    inputWithContext.set(chunk, contextSize);

    const feeds = {
      input: new ort.Tensor('float32', inputWithContext, [1, inputWithContext.length]),
      state: new ort.Tensor('float32', this.state, [2, 1, 128]),
      sr: new ort.Tensor('int64', BigInt64Array.from([BigInt(sr)])),
    };

    const outputNames = this.session.outputNames;
    const results = await this.session.run(feeds);
    const probTensor = results[outputNames[0]];
    const newStateTensor = results[outputNames[1]];

    this.state = newStateTensor.data.slice(); // keep a copy
    this.context = inputWithContext.slice(inputWithContext.length - contextSize);
    this.lastSr = sr;

    return probTensor.data[0];
  }
}

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

async function getSpeechTimestamps(
  inputPath,
  vad,
  {
    threshold = 0.5,
    minSpeechDurationMs = 250,
    minSilenceDurationMs = 100,
    speechPadMs = 30,
    returnSeconds = false,
    timeResolution = 1,
    negThreshold,
    sampleRate,
    returnMetadata = false,
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
  let leftover = Buffer.alloc(0);

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
      if (currentSpeech.end - currentSpeech.start > minSpeechSamples) {
        speeches.push(currentSpeech);
      }
      currentSpeech = {};
      triggered = false;
      tempEnd = 0;
    }
  };

  const streamDone = (async () => {
    for await (const chunk of ffmpeg.stdout) {
      const leftoverSamples = leftover.length / Float32Array.BYTES_PER_ELEMENT;
      const data = leftover.length ? Buffer.concat([leftover, chunk]) : chunk;
      const usableBytes = data.length - (data.length % 4);
      if (usableBytes > 0) {
        const floatData = new Float32Array(
          data.buffer,
          data.byteOffset,
          usableBytes / Float32Array.BYTES_PER_ELEMENT,
        );
        totalSamples += floatData.length - leftoverSamples;
        let offset = 0;
        while (offset + windowSize <= floatData.length) {
          const frame = floatData.subarray(offset, offset + windowSize);
          const curSample = processedSamples;
          processedSamples += windowSize;
          await processFrame(frame, curSample);
          offset += windowSize;
        }
        const remainingSamples = floatData.length - offset;
        if (remainingSamples > 0) {
          leftover = Buffer.from(
            data.buffer,
            data.byteOffset + offset * Float32Array.BYTES_PER_ELEMENT,
            remainingSamples * Float32Array.BYTES_PER_ELEMENT,
          );
        } else {
          leftover = Buffer.alloc(0);
        }
      } else {
        leftover = data;
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

  if (leftover.length) {
    const remaining = new Float32Array(
      leftover.buffer,
      leftover.byteOffset,
      leftover.length / Float32Array.BYTES_PER_ELEMENT,
    );
    const padded = new Float32Array(windowSize);
    padded.set(remaining);
    const curSample = processedSamples;
    await processFrame(padded, curSample);
    processedSamples += windowSize;
  }

  if (currentSpeech.start !== undefined) {
    currentSpeech.end = totalSamples;
    if (currentSpeech.end - currentSpeech.start > minSpeechSamples) {
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
  const result = returnSeconds
    ? speeches.map(({ start, end }) => ({
      start: convertSeconds(start),
      end: convertSeconds(end),
      startSample: start,
      endSample: end,
    }))
    : speeches.map(({ start, end }) => ({
      start,
      end,
      startSeconds: convertSeconds(start),
      endSeconds: convertSeconds(end),
    }));

  if (returnMetadata) {
    return { timestamps: result, totalSamples };
  }

  return result;
}

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
