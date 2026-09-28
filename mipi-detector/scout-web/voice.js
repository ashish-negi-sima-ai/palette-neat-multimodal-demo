"use strict";
/* The microphone is acquired only inside the Listen click handler. */
class ScoutVoice {
  constructor(settings, api, currentState) {
    this.settings = settings;
    this.api = api;
    this.currentState = currentState;
    this.clientId = crypto.randomUUID();
    this.epoch = 0;
    this.recording = false;
    this.starting = false;
    this.lastReady = null;
    this.audio = new Audio();
    this.audio.preload = 'auto';
    this.queue = [];
    this.el = id => document.getElementById(id);
    this.el('speech-panel').hidden = !settings.stt && !settings.tts;
    this.el('listen').hidden = !settings.stt;
    this.el('speech-hint').textContent = settings.stt
      ? 'Press Listen, describe the selected subject, then press Stop & inspect. Uses this device’s microphone.'
      : 'Spoken inspection replies are enabled. Speech recognition is disabled.';
    this.el('listen').onclick = () => this.toggle().catch(error => this.fail(error));
    this.el('cancel-speech').onclick = () => this.cancel();
    this.el('play-reply').onclick = () => this.play();
    this.audio.onended = () => { URL.revokeObjectURL(this.queue.shift()); this.play(); };
    this.audio.onerror = () => this.status('Could not play the reply. The written answer is in the timeline.');
    window.addEventListener('pagehide', () => {
      this.dispose();
      fetch('/api/speech/cancel', {method: 'POST', keepalive: true,
        headers: {'Content-Type': 'application/json'}, body: JSON.stringify({client_id: this.clientId})}).catch(() => {});
    });
  }
  status(text) { this.el('speech-status').textContent = text; }
  stopPlayback() {
    this.audio.pause(); this.audio.removeAttribute('src'); this.audio.load();
    for (const url of this.queue) URL.revokeObjectURL(url);
    this.queue = [];
    this.el('play-reply').hidden = true;
  }
  releaseMicrophone() {
    clearInterval(this.timer);
    this.stream?.getTracks().forEach(track => track.stop()); this.stream = null;
    this.input?.disconnect(); this.input = null;
    this.recorder?.disconnect(); this.recorder = null;
    this.context?.close().catch(() => {}); this.context = null;
  }
  dispose() {
    this.epoch++;
    this.releaseMicrophone(); this.stopPlayback();
    this.recording = this.starting = false; this.captured = null; this.chunks = [];
    this.el('listen').textContent = 'Listen'; this.el('listen').setAttribute('aria-pressed', 'false');
  }
  connectionLost() {
    if (!this.disconnected) this.dispose();
    this.disconnected = true;
    this.token = null;
    this.el('listen').disabled = true;
    this.status('Reconnecting to Modalix…');
  }
  async cancel() {
    this.dispose(); this.token = null;
    this.status('Speech cancelled.');
    try {
      this.cancelRequest = (this.cancelRequest || Promise.resolve()).catch(() => {}).then(() =>
        this.api('/api/speech/cancel', {client_id: this.clientId}));
      await this.cancelRequest;
    }
    catch (error) { this.status(error.message); }
  }
  async fail(error) { await this.cancel(); this.status(error.message || String(error)); }
  async toggle() {
    if (this.starting) return;
    if (this.recording || this.captured) {
      const epoch = this.epoch;
      this.el('listen').disabled = true;
      const blob = this.captured || await this.finishCapture();
      if (epoch !== this.epoch) return;
      this.captured = null;
      this.status('Transcribing on Modalix…');
      const response = await fetch(`/api/speech/recordings/${this.token}`, {
        method: 'POST', headers: {'Content-Type': 'audio/wav'}, body: blob});
      if (!response.ok) throw new Error((await response.json()).error);
      this.el('listen').textContent = 'Listen';
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
      throw new Error('Microphone capture needs an HTTPS browser with AudioWorklet support.');
    }
    this.stopPlayback();
    this.starting = true;
    const epoch = ++this.epoch;
    // Resume inside the user gesture, before awaiting HTTP or permissions.
    const context = new AudioContext(); this.context = context;
    await context.resume();
    this.status('Requesting microphone…');
    await this.cancelRequest;
    const begun = await this.api('/api/speech/begin', {client_id: this.clientId,
      object_class: this.el('object-class').value, zone: this.el('zone').checked});
    if (epoch !== this.epoch) {
      await this.api('/api/speech/cancel', {client_id: this.clientId}); return;
    }
    this.token = begun.id;
    this.generation = begun.mission_id;
    const stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1,
      echoCancellation: true, noiseSuppression: true}, video: false});
    if (epoch !== this.epoch) { stream.getTracks().forEach(track => track.stop()); return; }
    this.stream = stream;
    await context.audioWorklet.addModule('/audio-recorder.js');
    if (epoch !== this.epoch) return;
    this.rate = context.sampleRate; this.chunks = []; this.finishing = null;
    this.recorder = new AudioWorkletNode(context, 'scout-recorder', {
      processorOptions: {seconds: this.settings.record_seconds}});
    this.recorder.port.onmessage = event => {
      if (epoch !== this.epoch) return;
      if (event.data.samples) this.chunks.push(event.data.samples);
      if (event.data.done) this.flush?.();
      if (event.data.limit) {
        this.finishCapture().then(blob => {
          if (epoch !== this.epoch) return;
          this.captured = blob; this.el('listen').textContent = 'Submit recording';
          this.status('Recording limit reached; microphone off. Press Submit recording to inspect.');
        }).catch(error => this.fail(error));
      }
    };
    this.input = context.createMediaStreamSource(stream);
    this.input.connect(this.recorder); this.recorder.connect(context.destination);
    // The worklet emits silence: the microphone is never monitored on speakers.
    this.starting = false; this.recording = true; this.startedAt = performance.now();
    this.el('listen').textContent = 'Stop & inspect'; this.el('listen').setAttribute('aria-pressed', 'true');
    this.timer = setInterval(() => this.status(`Listening · ${Math.floor((performance.now() - this.startedAt) / 1000)} / ${this.settings.record_seconds}s`), 200);
    stream.getAudioTracks()[0].onended = () => { if (this.recording) this.fail(new Error('Microphone disconnected; recording cancelled.')); };
  }
  finishCapture() {
    if (!this.finishing) this.finishing = this.collectAudio();
    return this.finishing;
  }
  async collectAudio() {
    this.recording = false;
    this.el('listen').setAttribute('aria-pressed', 'false');
    if (this.recorder) {
      await new Promise(resolve => {
        const timeout = setTimeout(resolve, 1000);
        this.flush = () => { clearTimeout(timeout); resolve(); };
        this.recorder.port.postMessage('stop');
      });
    }
    const chunks = this.chunks; this.chunks = [];
    const rate = this.rate;
    this.releaseMicrophone();
    const source = new Float32Array(chunks.reduce((n, chunk) => n + chunk.length, 0));
    let offset = 0; for (const chunk of chunks) { source.set(chunk, offset); offset += chunk.length; }
    if (!source.length) throw new Error('No audio recorded. Please try again.');
    // The browser applies its audio resampler/antialiasing filter at the actual input rate.
    const count = Math.floor(source.length * 16000 / rate);
    const offline = new OfflineAudioContext(1, Math.max(1, count), 16000);
    const buffer = offline.createBuffer(1, source.length, rate); buffer.copyToChannel(source, 0);
    const node = offline.createBufferSource(); node.buffer = buffer; node.connect(offline.destination); node.start();
    const samples = (await offline.startRendering()).getChannelData(0);
    const wav = new ArrayBuffer(44 + samples.length * 2), view = new DataView(wav);
    const ascii = (at, text) => [...text].forEach((c, i) => view.setUint8(at + i, c.charCodeAt(0)));
    ascii(0, 'RIFF'); view.setUint32(4, wav.byteLength - 8, true); ascii(8, 'WAVEfmt ');
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, 16000, true); view.setUint32(28, 32000, true); view.setUint16(32, 2, true);
    view.setUint16(34, 16, true); ascii(36, 'data'); view.setUint32(40, samples.length * 2, true);
    samples.forEach((sample, i) => view.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, sample)) * (sample < 0 ? 32768 : 32767)), true));
    return new Blob([wav], {type: 'audio/wav'});
  }
  async play() {
    if (!this.queue.length) { this.status('Reply finished.'); this.el('play-reply').hidden = true; return; }
    if (this.recording || this.starting) return;
    this.audio.src = this.queue[0];
    try { await this.audio.play(); this.status('Speaking…'); this.el('play-reply').hidden = true; }
    catch (_) { this.status('Reply ready. Press Play reply to hear it.'); this.el('play-reply').hidden = false; }
  }
  async update(state, selectedCamera) {
    if (!this.settings.stt && !this.settings.tts) return;
    this.el('listen').hidden = !this.settings.stt || selectedCamera !== 'mipi';
    if (this.recording && (selectedCamera !== 'mipi' || state.mission_id > this.generation)) await this.cancel();
    if (this.polling) return;
    this.polling = true;
    const epoch = this.epoch;
    try {
      const speech = await this.api('/api/speech/state?client_id=' + encodeURIComponent(this.clientId));
      if (epoch !== this.epoch) return;
      if (this.disconnected || this.models !== speech.models) {
        this.disconnected = false;
        this.models = speech.models;
        if (!this.recording && !this.starting && !this.queue.length)
          this.status(speech.models === 'ready' ? 'Speech connected.' : 'Loading speech models…');
      }
      this.el('listen').disabled = this.starting || speech.models !== 'ready' || (speech.busy && !this.recording && !this.captured);
      this.el('cancel-speech').hidden = !(speech.id || this.recording || this.queue.length || this.starting);
      if (speech.id && speech.phase === 'error') {
        this.status(speech.error);
        if (speech.transcript && this.lastReady !== speech.id) {
          this.lastReady = speech.id;
          this.el('query').value = speech.transcript;
          this.status(speech.error + ' Edit the transcript, then inspect.');
        }
      }
      else if (speech.id && !this.recording && !this.captured && speech.busy) {
        const labels = {queued: 'Waiting for inference…', loading_stt: 'Loading Whisper…', transcribing: 'Transcribing…',
          loading_tts: 'Loading voice…', synthesizing: 'Preparing spoken reply…', cancelling: 'Discarding the current reply…'};
        this.status(labels[speech.phase] || speech.phase);
      }
      if (speech.id && speech.phase === 'cancelled' && (this.recording || this.captured)) await this.cancel();
      if (speech.id && speech.phase === 'cancelled' && this.queue.length) {
        this.stopPlayback(); this.status('Speech cancelled.');
      }
      if (speech.phase === 'ready' && speech.id && this.lastReady !== speech.id) {
        this.lastReady = speech.id;
        if (speech.transcript) this.status('Heard: ' + speech.transcript + '. Mission submitted.');
        if (speech.audio?.length) {
          this.stopPlayback();
          const urls = [];
          try {
            for (const url of speech.audio) {
              const response = await fetch(url);
              if (!response.ok) throw new Error('Spoken reply expired.');
              urls.push(URL.createObjectURL(await response.blob()));
            }
            if (epoch !== this.epoch) { urls.forEach(URL.revokeObjectURL); return; }
            this.queue = urls; await this.play();
          } catch (error) { urls.forEach(URL.revokeObjectURL); throw error; }
        }
      }
    } catch (error) {
      if (this.recording || this.starting) await this.cancel();
      this.status('Speech: ' + error.message);
    } finally { this.polling = false; }
  }
}
window.ScoutVoice = ScoutVoice;
