/* Collect actual AudioContext-rate mono samples; resampling happens after Stop. */
class ScoutRecorder extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.remaining = Math.floor(sampleRate * options.processorOptions.seconds);
    this.done = false;
    this.port.onmessage = event => {
      if (event.data === 'stop') { this.done = true; this.port.postMessage({done: true}); }
    };
  }
  process(inputs) {
    if (this.done) return true;
    const channels = inputs[0];
    if (!channels?.length) return true;
    const count = Math.min(channels[0].length, this.remaining);
    const mono = new Float32Array(count);
    for (const channel of channels) for (let i = 0; i < count; i++) mono[i] += channel[i] / channels.length;
    this.port.postMessage({samples: mono}, [mono.buffer]);
    this.remaining -= count;
    if (!this.remaining) { this.done = true; this.port.postMessage({limit: true}); }
    return true;
  }
}
registerProcessor('scout-recorder', ScoutRecorder);
