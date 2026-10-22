"""Export a standalone HTTP OpenAPI document with an explicit WebSocket extension."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sibi_bisindo.service.app import create_app
from sibi_bisindo.service.settings import Settings


def build_document():
    settings = Settings()
    document = create_app(settings).openapi()
    document['servers'] = [{'url': 'https://ai.signtech.tech', 'description': 'Production'}]
    document['security'] = []
    document['info']['description'] = (
        'Service pengenalan pose alfabet statis SIBI/BISINDO. Pilih bahasa secara eksplisit. '
        'Kirim byte JPEG/PNG, bukan JSON, base64, atau multipart. '
        'Kontrak HTTP menggunakan OpenAPI; detail webcam WebSocket tersedia pada '
        'extension x-websocket-contract di dokumen ini. Extension tersebut dibaca manual '
        'dan tidak otomatis menjadi request WebSocket pada importer OpenAPI. '
        'Tidak memerlukan API key/JWT saat ini. CORS bukan autentikasi.'
    )
    document['externalDocs'] = {'url': 'https://ai.signtech.tech/docs', 'description': 'Swagger UI live'}
    schemas = document['components']['schemas']
    schemas['ApiError'] = {
        'type': 'object', 'required': ['detail'],
        'properties': {'detail': {'type': 'string'}},
    }
    schemas['ModelDescription'] = {
        'type': 'object',
        'required': ['model_id', 'labels', 'recognition', 'preprocessing', 'limitations'],
        'properties': {
            'model_id': {'type': 'string'},
            'labels': {'type': 'array', 'items': {'type': 'string'}},
            'recognition': {'type': 'string', 'const': 'static_pose'},
            'preprocessing': {'type': 'string'},
            'limitations': {'type': 'array', 'items': {'type': 'string'}},
        },
    }
    selection = json.loads((ROOT / 'configs/service_models.json').read_text(encoding='utf-8'))
    examples = {}
    for language, path in selection['models'].items():
        example = dict(
            language=language, model_id=Path(path).name, status='predicted',
            detection_status='ok' if language == 'sibi' else 'two_hands_actor_unverified',
            label='A', score=0.94, score_type='softmax_uncalibrated',
            hand_count=1 if language == 'sibi' else 2, width=640, height=480,
            preprocessing=dict(method='baseline', attempts=0, recovered=False,
                               input_was_mirrored=False, coordinates='unmirrored_normalized_xyz'),
            timings_ms=dict(detector=30.0, classifier=1.0, total=32.0), landmarks=None,
        )
        examples[language] = {'summary': f'Contoh ilustratif {language.upper()}, bukan benchmark', 'value': example}
    no_hand = deepcopy(examples['sibi']['value'])
    no_hand.update(status='no_hand', detection_status='no_hand', label=None, score=None,
                   score_type=None, hand_count=0, landmarks=[])
    no_hand['timings_ms'] = dict(detector=30.0, classifier=0.0, total=31.0)
    examples['no_hand'] = {'summary': 'Frame valid, tidak ada tangan; include_landmarks=true', 'value': no_hand}

    prediction_properties = schemas['Prediction']['properties']
    descriptions = {
        'language': 'Model yang dipilih pemanggil; tidak ada pemilihan bahasa otomatis.',
        'status': 'predicted bila ekstraksi valid; selain itu no_hand, unexpected_extra_hand, atau invalid_geometry.',
        'detection_status': 'ok (SIBI satu tangan), one_hand_completeness_unverified, '
                            'two_hands_actor_unverified, no_hand, unexpected_extra_hand, atau invalid_geometry.',
        'label': 'Huruf hasil model. Null bila ekstraksi tidak valid. Tidak ada kelas unknown/background.',
        'score': 'Softmax belum dikalibrasi; 0.94 tidak menjamin 94% benar. Null bila tidak ada prediksi.',
        'hand_count': 'Jumlah tangan terdeteksi, tidak menjamin kelengkapan pose atau satu pelaku.',
        'landmarks': 'Null jika tidak diminta. Jika diminta: [jumlah_tangan][21][3] xyz; '
                     'tanpa tangan menghasilkan []. Slot diurutkan menurut wrist x lalu y pada gambar. '
                     'x/y ternormalisasi ukuran gambar; z relatif per tangan. Selalu koordinat unmirrored.',
        'timings_ms': 'Waktu proses server dalam milidetik, tidak mencakup perjalanan jaringan atau render browser.',
    }
    for name, description in descriptions.items():
        prediction_properties[name]['description'] = description

    for path, item in document['paths'].items():
        for method, operation in item.items():
            if method not in ('get', 'post'):
                continue
            for response in operation['responses'].values():
                response.setdefault('headers', {}).update({
                    'X-Request-ID': {'description': 'ID request untuk pelacakan galat HTTP.', 'schema': {'type': 'string'}},
                    'Cache-Control': {'schema': {'type': 'string', 'const': 'no-store'}},
                })
            for name in ('403', '500'):
                operation['responses'][name] = {
                    'description': 'Origin tidak diizinkan' if name == '403' else 'Galat internal; simpan error_id bila tersedia.',
                    'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ApiError'}}},
                }
    paths = document['paths']
    for endpoint, example in (
        ('/health/live', {'status': 'alive'}),
        ('/health/ready', {'status': 'ready', 'languages': ['sibi', 'bisindo']}),
    ):
        paths[endpoint]['get']['responses']['200']['content']['application/json'] = {
            'schema': {'type': 'object', 'required': list(example), 'properties': {
                'status': {'type': 'string', 'const': example['status']},
                **({'languages': {'type': 'array', 'items': {'type': 'string', 'enum': ['sibi', 'bisindo']}}}
                   if 'languages' in example else {}),
            }}, 'example': example,
        }
    paths['/health/ready']['get']['responses']['503'] = {
        'description': 'Model belum siap.',
        'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ApiError'}}},
    }
    paths['/v1/models']['get']['responses']['200']['content']['application/json']['schema'] = {
        'type': 'object', 'required': ['bundle_version', 'models'], 'properties': {
            'bundle_version': {'type': 'string'},
            'models': {'type': 'object', 'required': ['sibi', 'bisindo'], 'properties': {
                language: {'$ref': '#/components/schemas/ModelDescription'} for language in ('sibi', 'bisindo')
            }},
        },
    }
    predict = paths['/v1/predict/{language}']['post']
    predict['summary'] = 'Prediksi satu pose alfabet dari gambar'
    predict['description'] = 'Kirim isi berkas gambar secara langsung. Null label adalah hasil sah, bukan galat HTTP.'
    predict['responses']['200']['content']['application/json']['examples'] = examples
    for parameter in predict['parameters']:
        parameter['description'] = {
            'language': 'Pilih sibi atau bisindo secara eksplisit.',
            'include_landmarks': 'Sertakan titik landmark untuk overlay; default false.',
            'mirrored': 'true hanya jika piksel input sudah dimirror; default false. CSS mirror preview tidak memerlukan true.',
        }[parameter['name']]
    for code, description in {
        '413': 'Frame melampaui 2.000.000 byte.',
        '415': 'Content-Type harus image/jpeg atau image/png.',
        '503': 'Jalur inferensi sibuk atau service belum siap. Tunggu lalu kirim frame terbaru.',
    }.items():
        predict['responses'][code] = {
            'description': description,
            'content': {'application/json': {'schema': {'$ref': '#/components/schemas/ApiError'}}},
        }
    predict['responses']['503']['headers'] = {'Retry-After': {'schema': {'type': 'string'}, 'example': '1'}}
    predict['responses']['422'] = {
        'description': 'Validasi parameter gagal atau gambar kosong/rusak/dimensinya terlalu besar.',
        'content': {'application/json': {'schema': {'anyOf': [
            {'$ref': '#/components/schemas/ApiError'}, {'$ref': '#/components/schemas/HTTPValidationError'}
        ]}}},
    }
    predict['x-codeSamples'] = [{
        'lang': 'Shell', 'source': 'curl -X POST "https://ai.signtech.tech/v1/predict/sibi?include_landmarks=true" '
                                 '-H "Content-Type: image/jpeg" --data-binary @frame.jpg'
    }]
    schemas['WebSocketPrediction'] = {
        'allOf': [{'$ref': '#/components/schemas/Prediction'},
                  {'type': 'object', 'required': ['type'], 'properties': {'type': {'const': 'prediction', 'type': 'string'}}}],
    }
    schemas['WebSocketError'] = {
        'type': 'object', 'required': ['type', 'code', 'detail'], 'properties': {
            'type': {'type': 'string', 'const': 'error'}, 'code': {'type': 'string'}, 'detail': {'type': 'string'},
        },
    }
    document['x-websocket-contract'] = {
        'description': 'Kontrak tambahan untuk dibaca manual; bukan operasi HTTP OpenAPI.',
        'urls': {language: f'wss://ai.signtech.tech/v1/ws/{language}' for language in ('sibi', 'bisindo')},
        'query_parameters': [p for p in predict['parameters'] if p['in'] == 'query'],
        'handshake': 'Origin wajib dan harus ada di allowlist server. Browser mengirim Origin otomatis; '
                     'klien non-browser perlu mengisinya. Bahasa/origin tidak sah ditolak sebelum accept, '
                     'sehingga klien dapat melihat HTTP 403 saat handshake, bukan close frame 1008.',
        'send': {'frame_type': 'binary', 'payload': 'Byte JPEG/PNG berupa Blob atau ArrayBuffer. Bukan JSON/base64/multipart.'},
        'receive': {'frame_type': 'text', 'payload': 'JSON', 'schema': {'oneOf': [
            {'$ref': '#/components/schemas/WebSocketPrediction'}, {'$ref': '#/components/schemas/WebSocketError'}
        ]}},
        'flow': ['Buka koneksi dengan bahasa pilihan.', 'Ambil frame kamera dan encode JPEG/PNG.',
                 'Kirim satu frame biner lalu tunggu satu respons.',
                 'Saat type=prediction, periksa label null sebelum menampilkan huruf.',
                 'Saat code=503, tunggu singkat lalu kirim frame kamera terbaru.',
                 'Jangan menumpuk frame lama; jangan gunakan interval pengiriman tanpa menunggu respons.',
                 'Tutup koneksi lama sebelum mengganti bahasa; hentikan track kamera saat selesai.'],
        'error_codes': {'binary_frame_required': 'Frame teks ditolak.', 'empty_frame': 'Payload kosong.',
                        '422': 'Gambar tidak valid.', '503': 'Inferensi sibuk; retry dengan frame terbaru.',
                        '500': 'Galat internal; simpan error_id.'},
        'close_codes': {'1008': 'Idle tanpa frame selama 60 detik; kebijakan bahasa/origin juga memakai penolakan sebelum accept.',
                        '1009': 'Frame terlalu besar.'},
        'examples': {
            'prediction': {'type': 'prediction', **examples['sibi']['value']},
            'no_hand': {'type': 'prediction', **no_hand},
            'error': {'type': 'error', 'code': '503', 'detail': 'Inference lane is busy; retry after the current frame'},
        },
        'javascript_example': "const ws = new WebSocket('wss://ai.signtech.tech/v1/ws/sibi?include_landmarks=true');\n"
                              "// frameBlob harus JPEG/PNG; kirim sesudah event open.\n"
                              "ws.onopen = () => ws.send(frameBlob);\n"
                              "ws.onmessage = event => { const result = JSON.parse(event.data); console.log(result); };\n"
                              "// Kirim frame berikutnya hanya setelah onmessage. Gunakan examples/webcam.js untuk loop lengkap.",
    }
    document['x-integration'] = {
        'production_origins': ['https://signtech.tech', 'https://ai.signtech.tech'],
        'demo_origins': ['http://localhost:5173', 'http://127.0.0.1:5173'],
        'origin_note': 'Origin demo di atas telah diizinkan pada VM. www, subdomain, atau port lain perlu ditambahkan ke ALLOWED_ORIGINS.',
        'limits': {'max_frame_bytes': settings.max_frame_bytes, 'max_frame_pixels': settings.max_frame_pixels,
                   'websocket_idle_seconds': settings.websocket_idle_seconds, 'concurrent_inference': 1},
        'camera': {'suggested_resolution': [640, 480], 'jpeg_quality': 0.8, 'preserve_aspect_ratio': True,
                   'input_pixels': 'Unmirrored RGB encoded as JPEG/PNG.',
                   'overlay': 'Pasangkan landmark dengan frame yang dikirim; video live dapat bergerak lebih dulu daripada respons API.',
                   'secure_context': 'Browser memerlukan HTTPS atau localhost untuk akses webcam.'},
        'limitations': ['J/Z SIBI merupakan pose statis, belum pengenalan gerakan.',
                        'D/F/P BISINDO dapat tertukar pada webcam.',
                        'Jumlah tangan tidak menentukan bahasa; pilih mode secara eksplisit.',
                        'Skor tidak terkalibrasi dan tidak ada kelas unknown/background.'],
        'privacy': 'Service tidak menyimpan frame atau riwayat prediksi pengguna.',
        'authentication': 'Belum ada API key/JWT. Allowlist Origin bukan autentikasi.',
        'proxy_errors': 'Nginx dapat mengembalikan galat non-JSON, misalnya 502/504; periksa HTTP status sebelum parsing JSON.',
        'live_schema_url': 'https://ai.signtech.tech/openapi.json',
        'export_command': r'& .\.venv-service\Scripts\python.exe scripts/export_api_docs.py',
    }
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/api/openapi.json')
    args = parser.parse_args()
    document = build_document()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Exported {len(document["paths"])} HTTP paths + WebSocket contract: {args.output}')


if __name__ == '__main__':
    main()
