import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sibi_bisindo.service.app import create_app
from sibi_bisindo.service.settings import Settings
from sibi_bisindo.service.runtime import InferenceService, BadFrame, Busy
from sibi_bisindo.service.schemas import AutoPrediction


def response(language='sibi'):
    return dict(language=language,model_id='fixture',status='predicted',detection_status='ok',label='A',score=.8,
        score_type='softmax_uncalibrated',hand_count=1,width=640,height=480,landmarks=None,
        preprocessing=dict(method='baseline',attempts=0,recovered=False,input_was_mirrored=False,coordinates='unmirrored_normalized_xyz'),
        timings_ms=dict(detector=1.,classifier=.1,total=1.1))


class FakeService:
    def __init__(self,settings):self.ready=False;self.error=None;self.calls=[]
    async def start(self):self.ready=True
    async def close(self):self.ready=False
    def descriptions(self):return {'models':{'sibi':{},'bisindo':{}}}
    async def predict(self,*args):
        self.calls.append(args)
        if self.error:raise self.error
        if args[0]=='auto':return dict(response('sibi'),requested_language='auto',routing_hand_count=1)
        return response(args[0])


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.settings=Settings(max_frame_bytes=32)
        self.app=create_app(self.settings,FakeService)
        self.client=TestClient(self.app)
        self.client.__enter__()
    def tearDown(self):self.client.__exit__(None,None,None)

    def test_routes_lifespan_and_openapi_binary_body(self):
        self.assertEqual(self.client.get('/health/ready').status_code,200)
        self.assertIn('bisindo',self.client.get('/v1/models').json()['models'])
        schema=self.client.get('/openapi.json').json()
        self.assertIn('image/jpeg',schema['paths']['/v1/predict/{language}']['post']['requestBody']['content'])
        result=self.client.post('/v1/predict/bisindo?mirrored=true&include_landmarks=true',content=b'jpeg',headers={'Content-Type':'image/jpeg'})
        self.assertEqual(result.status_code,200)
        self.assertEqual(self.app.state.service.calls[-1],('bisindo',b'jpeg',True,True))
        self.assertEqual(len(result.headers['x-request-id']),32)
        self.assertEqual(result.headers['cache-control'],'no-store')

    def test_production_demo_serves_only_its_public_assets(self):
        home=self.client.get('/',follow_redirects=False)
        self.assertEqual(home.status_code,307)
        self.assertEqual(home.headers['location'],'/production.html')
        page=self.client.get('/production.html')
        self.assertEqual(page.status_code,200)
        self.assertIn('text/html',page.headers['content-type'])
        self.assertIn('<option value="auto">',page.text)
        for filename in ('webcam.css','webcam.js','production-demo.css','production-demo.js'):
            self.assertIn(filename,page.text)
            self.assertEqual(self.client.get('/'+filename).status_code,200)
        self.assertEqual(self.client.get('/webcam.html').status_code,404)
        self.assertNotIn('/production.html',self.client.get('/openapi.json').json()['paths'])

    def test_http_rejects_language_content_size_and_origin(self):
        cases=[('/v1/predict/asl',b'x',{'Content-Type':'image/jpeg'},422),
               ('/v1/predict/sibi',b'x',{'Content-Type':'application/json'},415),
               ('/v1/predict/sibi',b'',{'Content-Type':'image/jpeg'},422),
               ('/v1/predict/sibi',b'x'*33,{'Content-Type':'image/jpeg'},413),
               ('/v1/predict/sibi',b'x',{'Content-Type':'image/jpeg','Origin':'https://evil.example'},403)]
        for url,body,headers,status in cases:
            with self.subTest(status=status):self.assertEqual(self.client.post(url,content=body,headers=headers).status_code,status)
        self.assertEqual(self.app.state.service.calls,[])

    def test_auto_http_and_websocket_keep_manual_routes(self):
        schema=self.client.get('/openapi.json').json()
        self.assertIn('/v1/predict/auto',schema['paths'])
        headers={'Content-Type':'image/jpeg'}
        auto=self.client.post('/v1/predict/auto?include_landmarks=true',content=b'jpeg',headers=headers)
        self.assertEqual(auto.status_code,200)
        self.assertEqual((auto.json()['requested_language'],auto.json()['language'],auto.json()['routing_hand_count']),('auto','sibi',1))
        self.assertEqual(self.app.state.service.calls[-1],('auto',b'jpeg',True,False))
        manual=self.client.post('/v1/predict/bisindo',content=b'jpeg',headers=headers)
        self.assertEqual(manual.status_code,200)
        self.assertEqual(manual.json()['language'],'bisindo')
        with self.client.websocket_connect('/v1/ws/auto',headers={'Origin':self.settings.allowed_origins[0]}) as ws:
            ws.send_bytes(b'jpeg')
            result=ws.receive_json()
            self.assertEqual((result['type'],result['requested_language'],result['language']),('prediction','auto','sibi'))

    def test_cors_and_explicit_error_contract(self):
        origin=self.settings.allowed_origins[0]
        preflight=self.client.options('/v1/predict/sibi',headers={'Origin':origin,'Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'})
        self.assertEqual(preflight.status_code,200)
        self.assertEqual(preflight.headers['access-control-allow-origin'],origin)
        self.app.state.service.error=Busy('busy')
        result=self.client.post('/v1/predict/sibi',content=b'j',headers={'Content-Type':'image/jpeg'})
        self.assertEqual(result.status_code,503);self.assertEqual(result.headers['retry-after'],'1')
        self.app.state.service.error=BadFrame('invalid image')
        self.assertEqual(self.client.post('/v1/predict/sibi',content=b'j',headers={'Content-Type':'image/jpeg'}).status_code,422)

    def test_websocket_origin_binary_size_and_prediction(self):
        for origin in ['https://evil.example',None]:
            with self.assertRaises(WebSocketDisconnect) as error:
                headers={'Origin':origin} if origin else {}
                with self.client.websocket_connect('/v1/ws/sibi',headers=headers):pass
            self.assertEqual(error.exception.code,1008)
        with self.client.websocket_connect('/v1/ws/bisindo',headers={'Origin':self.settings.allowed_origins[0]}) as ws:
            ws.send_text('base64')
            self.assertEqual(ws.receive_json()['code'],'binary_frame_required')
            ws.send_bytes(b'jpeg')
            self.assertEqual(ws.receive_json()['type'],'prediction')
            self.app.state.service.error=Busy('busy')
            ws.send_bytes(b'jpeg');self.assertEqual(ws.receive_json()['code'],'503')
            ws.send_bytes(b'x'*33)
            with self.assertRaises(WebSocketDisconnect) as error:ws.receive_json()
            self.assertEqual(error.exception.code,1009)


class DecodeTests(unittest.TestCase):
    def test_decoder_rejects_bad_frames_limits_pixels_and_unmirrors_before_detection(self):
        service=InferenceService(Settings(max_frame_pixels=3))
        class Detector:
            def extract(self,rgb,*args,**kwargs):
                self.last=rgb.copy()
                return SimpleNamespace(valid=True,raw=np.zeros((1,21,3)),width=rgb.shape[1],height=rgb.shape[0],status='ok')
        class Model:
            def predict(self,*args):return {'label':'A','score':.8,'score_type':'softmax_uncalibrated'}
        detector=Detector();service.detectors={'sibi':detector};service.models={'sibi':Model()}
        service.bundle={'models':{'sibi':{'source_run':'fixture'}}}
        content=io.BytesIO();Image.fromarray(np.array([[[255,0,0],[0,0,255]]],dtype=np.uint8)).save(content,format='PNG')
        try:
            result=service._infer('sibi',content.getvalue(),False,True)
            self.assertEqual(result['label'],'A');np.testing.assert_array_equal(detector.last[0,0],[0,0,255])
            self.assertTrue(result['preprocessing']['input_was_mirrored'])
            with self.assertRaises(BadFrame):service._infer('sibi',b'bad',False,False)
            large=io.BytesIO();Image.new('RGB',(2,2)).save(large,format='PNG')
            with self.assertRaisesRegex(BadFrame,'dimensions'):service._infer('sibi',large.getvalue(),False,False)
        finally:service.executor.shutdown(wait=True)

    def test_auto_routes_by_detected_count_and_rejects_detector_disagreement(self):
        class Detector:
            def __init__(self, counts):self.counts=iter(counts);self.calls=0
            def extract(self,rgb,language,input_mirrored=False):
                count=next(self.counts);self.calls+=1
                status='no_hand' if count==0 else ('unexpected_extra_hand' if language=='sibi' and count==2 else 'ok')
                return SimpleNamespace(raw=np.zeros((count,21,3),dtype=np.float32),valid=count>0 and status=='ok',
                                       status=status,width=rgb.shape[1],height=rgb.shape[0])
        class Model:
            def __init__(self,label):self.label=label;self.calls=0
            def predict(self,*args):self.calls+=1;return {'label':self.label,'score':.8,'score_type':'softmax_uncalibrated'}
        service=InferenceService(Settings())
        service.detectors={'sibi':Detector([0,1,2,2]),'bisindo':Detector([2,1])}
        service.models={'sibi':Model('S'),'bisindo':Model('B')}
        service.bundle={'models':{'sibi':{'source_run':'sibi-run'},'bisindo':{'source_run':'bisindo-run'}}}
        content=io.BytesIO();Image.new('RGB',(2,2)).save(content,format='PNG')
        try:
            no_hand=service._infer_auto(content.getvalue(),True,False)
            self.assertEqual((no_hand['language'],no_hand['label'],no_hand['routing_hand_count']),(None,None,0))
            self.assertEqual(no_hand['landmarks'],[])
            AutoPrediction.model_validate(no_hand)
            one=service._infer_auto(content.getvalue(),False,False)
            self.assertEqual((one['language'],one['label'],one['hand_count']),('sibi','S',1))
            two=service._infer_auto(content.getvalue(),False,False)
            self.assertEqual((two['language'],two['label'],two['hand_count']),('bisindo','B',2))
            mismatch=service._infer_auto(content.getvalue(),False,False)
            self.assertEqual((mismatch['status'],mismatch['language'],mismatch['label'],mismatch['routing_hand_count'],mismatch['hand_count']),
                             ('route_disagreement',None,None,2,1))
            AutoPrediction.model_validate(mismatch)
            self.assertEqual((service.models['sibi'].calls,service.models['bisindo'].calls),(1,1))
            self.assertEqual((service.detectors['sibi'].calls,service.detectors['bisindo'].calls),(4,2))
        finally:service.executor.shutdown(wait=True)


if __name__=='__main__':unittest.main()
