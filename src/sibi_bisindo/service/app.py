"""FastAPI contract: one encoded image in, one alphabet prediction out."""
import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from .settings import Settings
from .runtime import InferenceService, BadFrame, Busy
from .schemas import Prediction

log = logging.getLogger(__name__)
Language = Literal['sibi','bisindo']


def create_app(settings=None, service_factory=InferenceService):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        service = service_factory(settings)
        await service.start()
        app.state.service = service
        try: yield
        finally: await service.close()

    app = FastAPI(title='SIBI / BISINDO Alphabet API',version='1.0.0',lifespan=lifespan,
        description='Static alphabet poses. Send unmirrored RGB webcam images encoded as JPEG/PNG; scores are uncalibrated.')
    app.add_middleware(CORSMiddleware,allow_origins=list(settings.allowed_origins),allow_credentials=False,
        allow_methods=['GET','POST'],allow_headers=['Content-Type'],expose_headers=['X-Request-ID'])

    @app.middleware('http')
    async def origin_and_request_id(request, call_next):
        request_id = uuid4().hex
        if request.headers.get('origin') and request.headers['origin'] not in settings.allowed_origins:
            response = JSONResponse({'detail':'Origin is not allowed'},status_code=403)
        else:
            response = await call_next(request)
        response.headers['X-Request-ID'] = request_id
        response.headers['Cache-Control'] = 'no-store'
        return response

    # Only the five files needed for the public webcam demo are served here.
    demo_dir = Path(__file__).resolve().parents[3] / 'examples'
    for filename in ('production.html', 'webcam.css', 'webcam.js',
                     'production-demo.css', 'production-demo.js'):
        def demo_asset(name):
            async def serve(request: Request):
                return FileResponse(demo_dir / name)
            return serve
        app.add_route('/' + filename, demo_asset(filename), methods=['GET'])

    @app.get('/', include_in_schema=False)
    async def homepage(): return RedirectResponse('/production.html')

    @app.get('/health/live',tags=['health'])
    async def live(): return {'status':'alive'}

    @app.get('/health/ready',tags=['health'])
    async def ready(request: Request):
        if not request.app.state.service.ready: raise HTTPException(503,'Models are not ready')
        return {'status':'ready','languages':['sibi','bisindo']}

    @app.get('/v1/models',tags=['models'])
    async def models(request: Request): return request.app.state.service.descriptions()

    async def run_prediction(service,language,payload,include_landmarks,mirrored):
        try:
            response=await service.predict(language,payload,include_landmarks,mirrored)
            return Prediction.model_validate(response).model_dump()
        except BadFrame as exc: raise HTTPException(422,str(exc)) from exc
        except Busy as exc: raise HTTPException(503,str(exc),headers={'Retry-After':'1'}) from exc
        except Exception as exc:
            error_id=uuid4().hex
            log.exception('Inference failed, error_id=%s',error_id)
            raise HTTPException(500,f'Inference failed; error_id={error_id}') from exc

    @app.post('/v1/predict/{language}',tags=['prediction'],response_model=Prediction,openapi_extra={
        'requestBody':{'required':True,'content':{kind:{'schema':{'type':'string','format':'binary'}}
                                                for kind in ('image/jpeg','image/png')}}})
    async def predict(language: Language,request: Request,include_landmarks: bool=False,mirrored: bool=False):
        if request.headers.get('content-type','').split(';')[0].strip().lower() not in {'image/jpeg','image/png'}:
            raise HTTPException(415,'Content-Type must be image/jpeg or image/png; send raw image bytes')
        payload=bytearray()
        async for part in request.stream():
            if len(payload)+len(part)>settings.max_frame_bytes: raise HTTPException(413,'Frame exceeds MAX_FRAME_BYTES')
            payload.extend(part)
        if not payload: raise HTTPException(422,'Frame is empty')
        return await run_prediction(request.app.state.service,language,bytes(payload),include_landmarks,mirrored)

    @app.websocket('/v1/ws/{language}')
    async def webcam(socket: WebSocket,language: str,include_landmarks: bool=False,mirrored: bool=False):
        # CORS middleware does not cover WebSocket handshakes.
        if language not in ('sibi','bisindo') or socket.headers.get('origin') not in settings.allowed_origins:
            await socket.close(code=1008)
            return
        await socket.accept()
        try:
            while True:
                try: message=await asyncio.wait_for(socket.receive(),timeout=settings.websocket_idle_seconds)
                except TimeoutError:
                    await socket.close(code=1008,reason='Idle timeout');return
                if message['type']=='websocket.disconnect': return
                payload=message.get('bytes')
                if payload is None:
                    await socket.send_json({'type':'error','code':'binary_frame_required','detail':'Send JPEG/PNG bytes, not base64 or JSON'})
                    continue
                if len(payload)>settings.max_frame_bytes:
                    await socket.close(code=1009,reason='Frame too large');return
                if not payload:
                    await socket.send_json({'type':'error','code':'empty_frame','detail':'Frame is empty'});continue
                try:
                    response=await run_prediction(socket.app.state.service,language,payload,include_landmarks,mirrored)
                    await socket.send_json({'type':'prediction',**response})
                except HTTPException as exc:
                    await socket.send_json({'type':'error','code':str(exc.status_code),'detail':exc.detail})
        except WebSocketDisconnect:
            return

    return app
