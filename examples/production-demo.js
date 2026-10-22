'use strict';
(() => {
  const element = id => document.getElementById(id);
  const base = document.body.dataset.apiBase;
  const report = element('production-report');
  const button = element('production-check');
  const add = (message, state = '') => {
    const row = document.createElement('li'); row.textContent = message; row.className = state;
    report.append(row);
  };
  async function request(path, options = {}) {
    const response = await fetch(base + path, { ...options, signal: AbortSignal.timeout(30000) });
    if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    return response.json();
  }
  function checkSocket(language, blob) {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(base.replace(/^https:/, 'wss:') + '/v1/ws/' + language);
      const timeout = setTimeout(() => finish(new Error('WebSocket tidak merespons dalam 30 detik.')), 30000);
      function finish(error, value) {
        clearTimeout(timeout); socket.onclose = null; socket.onerror = null;
        socket.onopen = null; socket.onmessage = null; socket.close();
        if (error) reject(error); else resolve(value);
      }
      socket.onopen = () => socket.send(blob);
      socket.onmessage = event => {
        try {
          const value = JSON.parse(event.data);
          if (value.type !== 'prediction') throw new Error(value.detail || 'Format respons tidak sesuai.');
          finish(null, value);
        } catch (error) { finish(error); }
      };
      socket.onerror = () => finish(new Error('WebSocket gagal: periksa koneksi dan allowlist Origin.'));
      socket.onclose = () => finish(new Error('WebSocket ditutup sebelum respons.'));
    });
  }
  function assertBlank(value, language) {
    if (value.language !== language || value.detection_status !== 'no_hand' || value.label !== null || value.hand_count !== 0) {
      throw new Error('Respons gambar kosong tidak sesuai kontrak: ' + JSON.stringify(value));
    }
  }
  button.onclick = async () => {
    report.replaceChildren(); element('production-result').textContent = 'Menguji API produksi...';
    element('production-report-panel').dataset.result = 'running';
    const controls = ['start', 'language', 'check-api', 'production-check'];
    controls.forEach(id => { element(id).disabled = true; });
    try {
      add('Origin browser: ' + location.origin);
      const ready = await request('/health/ready');
      if (ready.status !== 'ready') throw new Error('Model belum siap.');
      add('HTTPS + CORS: API ready', 'passed');
      const models = await request('/v1/models');
      for (const language of ['sibi', 'bisindo']) {
        if (!models.models?.[language]) throw new Error('Metadata model tidak lengkap.');
        add(`${language.toUpperCase()}: ${models.models[language].model_id}`, 'passed');
      }
      const canvas = document.createElement('canvas'); canvas.width = 640; canvas.height = 480;
      const ctx = canvas.getContext('2d'); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, 640, 480);
      const blank = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', .8));
      if (!blank) throw new Error('Browser gagal membuat gambar uji.');
      for (const language of ['sibi', 'bisindo']) {
        const http = await request('/v1/predict/' + language, {
          method: 'POST', headers: { 'Content-Type': 'image/jpeg' }, body: blank,
        });
        assertBlank(http, language); add(`HTTP ${language.toUpperCase()}: no_hand (sesuai gambar kosong)`, 'passed');
        const ws = await checkSocket(language, blank);
        assertBlank(ws, language); add(`WSS ${language.toUpperCase()}: no_hand (sesuai gambar kosong)`, 'passed');
      }
      element('production-result').textContent = 'Lulus: HTTPS, CORS, HTTP, dan WSS kedua model. Klik Mulai kamera untuk menguji pose.';
      element('production-report-panel').dataset.result = 'passed';
    } catch (error) {
      add(error.message || String(error), 'error');
      element('production-result').textContent = 'Uji belum lulus. Periksa detail di bawah dan izin origin pada server.';
      element('production-report-panel').dataset.result = 'failed';
    } finally { controls.forEach(id => { element(id).disabled = false; }); }
  };
})();
