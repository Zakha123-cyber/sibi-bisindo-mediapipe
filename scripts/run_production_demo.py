"""Serve the HTML webcam demo locally; all inference goes directly to production."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


class DemoHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(302)
            self.send_header('Location', '/production.html')
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=5173)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), partial(DemoHandler, directory=str(EXAMPLES)))
    except OSError:
        raise SystemExit(f'Port {args.port} tidak tersedia. Hentikan server lokal lain yang memakai port ini.')
    print(f'Buka http://localhost:{args.port}/production.html', flush=True)
    print('Inferensi: https://ai.signtech.tech. Server lokal hanya menyajikan HTML/CSS/JS.', flush=True)
    if args.port != 5173:
        print('Port berbeda memerlukan origin tambahan pada ALLOWED_ORIGINS server produksi.', flush=True)
    print('Ctrl+C untuk berhenti.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
