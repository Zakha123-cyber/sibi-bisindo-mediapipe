'use strict';
const $ = id => document.getElementById(id);
if (document.body.dataset.apiBase) $('api').value = document.body.dataset.apiBase;
else if (location.pathname.startsWith('/demo/')) $('api').value = location.origin;
let session = 0, stream = null, socket = null, timer = null, watchdog = null;
let active = false, pending = false, lastLandmarks = [], responses = [];
let capturedAt = 0, hasProcessedFrame = false;

function badge(text, state = '') { $('connection').textContent = text; $('connection').className = `badge ${state}`; }
function status(text, error = false) { $('status').textContent = text; $('status').classList.toggle('error', error); }
function clearOverlay() { const c = $('overlay'); c.getContext('2d').clearRect(0, 0, c.width, c.height); }
function clearPrediction(text = 'Menunggu pose tangan') {
  $('letter').textContent = '—'; $('prediction-state').textContent = text; $('score').textContent = '—';
  $('score-fill').style.width = '0%'; $('score-meter').setAttribute('aria-valuenow', '0');
  $('hands').textContent = '—'; $('latency').textContent = '—'; $('frame-age').textContent = '—'; $('fps').textContent = '—';
  lastLandmarks = []; clearOverlay();
}
function resetControls() {
  $('start').disabled = active; $('stop').disabled = !active;
  for (const id of ['api', 'language', 'check-api']) $(id).disabled = active;
  if ($('production-check')) $('production-check').disabled = active;
}
function stop(message = 'Kamera dihentikan.', error = false) {
  session++; active = false; pending = false; clearTimeout(timer); clearTimeout(watchdog);
  if (socket) { socket.onclose = null; socket.onerror = null; socket.close(); socket = null; }
  if (stream) { stream.getTracks().forEach(track => { track.onended = null; track.stop(); }); stream = null; }
  $('video').srcObject = null; $('placeholder').hidden = false; $('camera-state').textContent = 'Belum aktif';
  hasProcessedFrame = false; $('stage').classList.remove('synced');
  responses = []; clearPrediction('Kamera berhenti'); resetControls(); status(message, error);
  badge(error ? 'Koneksi terhenti' : 'Kamera berhenti', error ? 'error' : '');
}
function fail(message) { stop(message, true); }
function baseUrl() {
  const url = new URL($('api').value);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) throw new Error('Alamat API harus menggunakan http:// atau https://.');
  url.search = ''; url.hash = ''; url.pathname = url.pathname.replace(/\/+$/, '');
  return url.toString().replace(/\/$/, '');
}
async function apiRequest(path) {
  const controller = new AbortController(); const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch(baseUrl() + path, { signal: controller.signal, cache: 'no-store' });
    if (!response.ok) throw new Error(response.status === 403 ? 'Origin halaman belum diizinkan oleh API.' : `API merespons HTTP ${response.status}.`);
    return await response.json();
  } finally { clearTimeout(timeout); }
}
async function checkApi() {
  badge('Memeriksa API…');
  const result = await apiRequest('/health/ready');
  if (result.status !== 'ready') throw new Error('Model pada API belum siap.');
  badge('API siap', 'ready'); return result;
}
function friendlyError(error) {
  if (error.name === 'NotAllowedError') return 'Izin kamera ditolak. Izinkan kamera pada pengaturan browser lalu coba lagi.';
  if (error.name === 'NotFoundError') return 'Kamera tidak ditemukan pada perangkat ini.';
  if (error.name === 'NotReadableError') return 'Kamera sedang dipakai aplikasi lain. Tutup aplikasi kamera lalu coba lagi.';
  if (error.name === 'AbortError') return 'API belum merespons. Tunggu startup model selesai di terminal.';
  if (error instanceof TypeError) return document.body.dataset.apiBase
    ? `API produksi belum terhubung. Periksa internet dan izin origin ${location.origin} pada server.`
    : 'API belum terhubung. Jalankan scripts/run_local_demo.py dan periksa alamat API.';
  return error.message || 'Terjadi kesalahan. Coba hubungkan kembali.';
}
function drawLandmarks() {
  clearOverlay(); if (!$('landmarks').checked) return;
  const canvas = $('overlay'), ctx = canvas.getContext('2d');
  const chains = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20],[5,9,13,17]];
  lastLandmarks.forEach((hand, index) => {
    if (hand.length !== 21 || hand.some(p => p.length < 2 || !Number.isFinite(p[0]) || !Number.isFinite(p[1]))) return;
    const points = hand.map(p => [p[0] * canvas.width, p[1] * canvas.height]);
    ctx.strokeStyle = index === 0 ? '#62ebca' : '#ffd27a'; ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = 2;
    for (const chain of chains) {
      ctx.beginPath(); ctx.moveTo(...points[chain[0]]); chain.slice(1).forEach(i => ctx.lineTo(...points[i])); ctx.stroke();
    }
    for (const point of points) { ctx.beginPath(); ctx.arc(...point, 3, 0, Math.PI * 2); ctx.fill(); }
  });
}
function schedule(token, delay = 0) { clearTimeout(timer); timer = setTimeout(() => sendFrame(token), delay); }
function sendFrame(token) {
  if (token !== session || !active || pending || !socket || socket.readyState !== WebSocket.OPEN) return;
  const video = $('video'), canvas = $('capture');
  if (!video.videoWidth) { schedule(token, 100); return; }
  canvas.width = Math.min(640, video.videoWidth); canvas.height = Math.round(video.videoHeight * canvas.width / video.videoWidth);
  capturedAt = performance.now();
  const context = canvas.getContext('2d');
  context.filter = $('low-light').checked ? 'brightness(1.25) contrast(1.10)' : 'none';
  context.drawImage(video, 0, 0, canvas.width, canvas.height); // Keep inference unmirrored; filter the pixels sent to the API.
  context.filter = 'none';
  pending = true;
  canvas.toBlob(blob => {
    if (token !== session || !active) return;
    if (!blob || !socket || socket.readyState !== WebSocket.OPEN) { fail('Frame tidak dapat dikirim. Hubungkan kamera kembali.'); return; }
    try { socket.send(blob); } catch (error) { fail(friendlyError(error)); return; }
    watchdog = setTimeout(() => { if (token === session) fail('Respons frame terlalu lama. Periksa terminal API.'); }, 15000);
  }, 'image/jpeg', .8);
}
function showResult(result, frameAge) {
  $('letter').textContent = result.label || '—';
  const chosen = $('language').value === 'auto' ? `Mode ${result.language?.toUpperCase() || 'belum dipilih'} · ` : '';
  $('prediction-state').textContent = chosen + (result.label ? `Pose huruf ${result.label}` : 'Belum ada huruf');
  $('score').textContent = result.score == null ? '—' : result.score.toFixed(3);
  const score = Math.max(0, Math.min(1, result.score || 0));
  $('score-fill').style.width = `${score * 100}%`; $('score-meter').setAttribute('aria-valuenow', String(score));
  $('hands').textContent = result.routing_hand_count != null && result.routing_hand_count !== result.hand_count
    ? `${result.hand_count} (awal ${result.routing_hand_count})` : String(result.hand_count);
  $('latency').textContent = `${Math.round(result.timings_ms.total)} ms`;
  $('frame-age').textContent = `${Math.round(frameAge)} ms`;
  responses.push(performance.now()); responses = responses.slice(-12);
  $('fps').textContent = responses.length < 2 ? '—' : ((responses.length - 1) * 1000 / (responses.at(-1) - responses[0])).toFixed(1);
  const messages = { no_hand: 'Belum ada tangan terlihat. Arahkan tangan ke kamera.', unexpected_extra_hand: 'SIBI memakai satu tangan. Jauhkan tangan kedua.', invalid_geometry: 'Landmark belum valid. Perjelas posisi tangan.', route_disagreement: 'Jumlah tangan berbeda antar detektor. Tahan pose dan pastikan kedua tangan terlihat jelas.' };
  status(result.label ? 'Tahan pose sebentar dan amati hasil hurufnya.' : (messages[result.detection_status] || 'Pose belum dapat diprediksi.'));
  lastLandmarks = result.landmarks || []; drawLandmarks();
}
function showProcessedFrame() {
  const source = $('capture'), target = $('processed-frame');
  target.width = source.width; target.height = source.height;
  target.getContext('2d').drawImage(source, 0, 0);
  hasProcessedFrame = true;
  $('stage').classList.toggle('synced', $('sync-preview').checked);
}
async function start() {
  const token = ++session; active = true; hasProcessedFrame = false; $('stage').classList.remove('synced');
  resetControls(); clearPrediction(); status('Memeriksa API dan menyiapkan kamera…');
  try {
    await checkApi(); if (token !== session) return;
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) throw new Error('Buka halaman melalui localhost atau HTTPS agar kamera dapat diakses.');
    const camera = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
    if (token !== session) { camera.getTracks().forEach(track => track.stop()); return; }
    stream = camera; stream.getVideoTracks()[0].onended = () => { if (token === session) fail('Kamera terputus dari perangkat.'); };
    $('video').srcObject = stream; await $('video').play(); if (token !== session) return;
    $('stage').style.aspectRatio = `${$('video').videoWidth} / ${$('video').videoHeight}`;
    $('overlay').width = $('video').videoWidth; $('overlay').height = $('video').videoHeight;
    $('placeholder').hidden = true; $('camera-state').textContent = 'Kamera aktif';
    const url = new URL(baseUrl()); url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    url.pathname = url.pathname.replace(/\/$/, '') + '/v1/ws/' + $('language').value; url.search = '?include_landmarks=true';
    const connection = new WebSocket(url); socket = connection;
    watchdog = setTimeout(() => { if (token === session) fail('WebSocket belum tersambung. Periksa API dan origin halaman.'); }, 10000);
    connection.onopen = () => { if (token !== session) return; clearTimeout(watchdog); badge('Tersambung', 'ready'); status('Bentuk pose huruf di depan kamera.'); sendFrame(token); };
    connection.onmessage = event => {
      if (token !== session) return; clearTimeout(watchdog); pending = false;
      try {
        const result = JSON.parse(event.data);
        if (result.type === 'error') {
          clearPrediction('Menunggu API'); status(result.detail, true); schedule(token, 1000); return;
        }
        if (result.type !== 'prediction') throw new Error('Format respons API tidak dikenali.');
        const frameAge = performance.now() - capturedAt;
        showProcessedFrame(); showResult(result, frameAge); schedule(token);
      } catch (error) { fail(friendlyError(error)); }
    };
    connection.onerror = () => { if (token === session) fail('WebSocket gagal. Periksa alamat API dan origin halaman yang diizinkan.'); };
    connection.onclose = () => { if (token === session) fail('Koneksi API ditutup. Klik Mulai kamera untuk menghubungkan kembali.'); };
  } catch (error) { if (token === session) fail(friendlyError(error)); }
}
function languageHelp() {
  const language = $('language').value;
  $('pose-help').textContent = language === 'sibi' ? 'Gunakan satu tangan dan pastikan semua jari terlihat.'
    : language === 'bisindo' ? 'Bentuk pose BISINDO; kedua tangan masuk kamera bila pose membutuhkannya.'
    : 'Otomatis: satu tangan diarahkan ke SIBI, dua tangan ke BISINDO. Tahan pose agar kedua tangan terlihat.';
  $('limitations').textContent = language === 'sibi' ? 'J dan Z SIBI dikenali sebagai pose statis, belum sebagai gerakan.'
    : language === 'bisindo' ? 'Huruf D, F, dan P BISINDO masih dapat tertukar pada webcam.'
    : 'Mode otomatis mengikuti jumlah tangan yang terdeteksi per frame; jumlah tangan belum membuktikan bahasanya.';
  clearPrediction('Menunggu kamera');
}
$('start').onclick = start; $('stop').onclick = () => stop();
$('check-api').onclick = async () => { $('check-api').disabled = true; try { await checkApi(); status('API siap. Klik Mulai kamera.'); } catch (error) { badge('API belum terhubung', 'error'); status(friendlyError(error), true); } finally { $('check-api').disabled = active; } };
$('language').onchange = languageHelp;
$('mirror').onchange = () => $('stage').classList.toggle('mirrored', $('mirror').checked);
$('landmarks').onchange = drawLandmarks;
$('sync-preview').onchange = () => $('stage').classList.toggle('synced', active && hasProcessedFrame && $('sync-preview').checked);
window.addEventListener('beforeunload', () => stop());
languageHelp();
