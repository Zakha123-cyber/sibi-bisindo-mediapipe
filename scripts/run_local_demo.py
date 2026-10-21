"""Run the API and a simple browser webcam page together, without Docker."""
import argparse
from dataclasses import replace
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def create_local_app(port=8000):
    from fastapi.responses import RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from sibi_bisindo.service.app import create_app
    from sibi_bisindo.service.settings import Settings
    settings = Settings.from_env()
    bundle = settings.bundle_dir if settings.bundle_dir.is_absolute() else ROOT / settings.bundle_dir
    settings = replace(settings, bundle_dir=bundle, allowed_origins=tuple(dict.fromkeys([
        *settings.allowed_origins, f'http://localhost:{port}', f'http://127.0.0.1:{port}'])))
    app = create_app(settings)
    app.mount('/demo', StaticFiles(directory=ROOT / 'examples'), name='local-demo')

    @app.get('/', include_in_schema=False)
    async def home(): return RedirectResponse('/demo/webcam.html')

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535: parser.error('Port must be between 1 and 65535')
    with socket.socket() as probe:
        try: probe.bind(('127.0.0.1', args.port))
        except OSError:
            raise SystemExit(f'Port {args.port} sedang dipakai. Hentikan API lama atau gunakan --port PORT_LAIN.')
    import uvicorn
    print(f'Buka http://localhost:{args.port}/demo/webcam.html setelah startup model selesai.', flush=True)
    print('API dan halaman berjalan bersama. Ctrl+C untuk berhenti; kamera dibuka oleh browser saat Anda menekan Mulai.', flush=True)
    uvicorn.run(create_local_app(args.port), host='127.0.0.1', port=args.port, workers=1,
        ws_max_size=2_000_000, ws_max_queue=1, limit_concurrency=64, access_log=False)


if __name__ == '__main__': main()
