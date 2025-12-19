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
  const session = await ort.InferenceSession.create(modelPath, opts.sessionOptions);
  const vad = new SileroVad(session);
  vad.defaultSampleRate = spec ? spec.sampleRate : 16000;
  return vad;
}

async function getSpeechTimestamps(
  audio,
  vad,
  {
    samplingRate = 16000,
    threshold = 0.5,
    minSpeechDurationMs = 250,
    minSilenceDurationMs = 100,
    speechPadMs = 30,
    returnSeconds = false,
    timeResolution = 1,
    negThreshold,
  } = {},
) {
  if (!vad) {
    throw new Error('Pass a loaded SileroVad instance');
  }

  let sr = samplingRate;
  let wav = audio instanceof Float32Array ? audio : Float32Array.from(audio);

  // Downsample if the audio is a multiple of 16 kHz (mirrors the Python helper).
  if (sr > 16000 && sr % 16000 === 0) {
    const step = sr / 16000;
    const reduced = new Float32Array(Math.ceil(wav.length / step));
    for (let i = 0, j = 0; i < wav.length; i += step, j += 1) {
      reduced[j] = wav[Math.floor(i)];
    }
    wav = reduced;
    sr = 16000;
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

  for (let i = 0; i < wav.length; i += windowSize) {
    const frame = wav.subarray(i, i + windowSize);
    const padded =
      frame.length === windowSize
        ? frame
        : new Float32Array([...frame, ...new Float32Array(windowSize - frame.length)]);

    const speechProb = await vad.processChunk(padded, sr);
    const curSample = i;

    if (speechProb >= threshold && tempEnd) {
      tempEnd = 0;
    }

    if (speechProb >= threshold && !triggered) {
      triggered = true;
      currentSpeech.start = curSample;
      continue;
    }

    if (speechProb < negThres && triggered) {
      if (!tempEnd) {
        tempEnd = curSample;
      }
      if (curSample - tempEnd < minSilenceSamples) {
        continue;
      }

      currentSpeech.end = tempEnd;
      if (currentSpeech.end - currentSpeech.start > minSpeechSamples) {
        speeches.push(currentSpeech);
      }
      currentSpeech = {};
      triggered = false;
      tempEnd = 0;
    }
  }

  if (currentSpeech.start !== undefined) {
    currentSpeech.end = wav.length;
    if (currentSpeech.end - currentSpeech.start > minSpeechSamples) {
      speeches.push(currentSpeech);
    }
  }

  // Pad segments a bit for nicer cuts.
  for (let idx = 0; idx < speeches.length; idx += 1) {
    const speech = speeches[idx];
    const prevEnd = idx === 0 ? 0 : speeches[idx - 1].end;
    const nextStart = idx === speeches.length - 1 ? wav.length : speeches[idx + 1].start;
    const padStart = Math.max(speech.start - speechPadSamples, prevEnd);
    const padEnd = Math.min(speech.end + speechPadSamples, nextStart);
    speech.start = Math.max(0, Math.floor(padStart));
    speech.end = Math.min(wav.length, Math.floor(padEnd));
  }

  if (returnSeconds) {
    const convert = (samples) => +(samples / sr).toFixed(timeResolution);
    return speeches.map(({ start, end }) => ({ start: convert(start), end: convert(end) }));
  }

  return speeches;
}

// Decode arbitrary audio with ffmpeg into mono, 16 kHz, float32 PCM for the VAD.
function decodeWithFfmpeg(inputPath, { sampleRate = 16000, channels = 1 } = {}) {
  return new Promise((resolve, reject) => {
    const args = [
      '-v',
      'error',
      '-i',
      inputPath,
      '-ac',
      String(channels),
      '-ar',
      String(sampleRate),
      '-f',
      'f32le',
      'pipe:1',
    ];
    const ffmpeg = spawn('ffmpeg', args, { stdio: ['ignore', 'pipe', 'inherit'] });
    const chunks = [];

    ffmpeg.stdout.on('data', (data) => chunks.push(data));
    ffmpeg.on('error', reject);
    ffmpeg.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`ffmpeg exited with code ${code}`));
        return;
      }
      const buffer = Buffer.concat(chunks);
      const floatData = new Float32Array(
        buffer.buffer,
        buffer.byteOffset,
        buffer.byteLength / Float32Array.BYTES_PER_ELEMENT,
      );
      resolve(floatData);
    });
  });
}

module.exports = {
  loadSileroVad,
  getSpeechTimestamps,
  decodeWithFfmpeg,
  WEIGHTS,
};
