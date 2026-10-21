"""One serialized CPU inference lane; never share a MediaPipe graph concurrently."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class BadFrame(ValueError):
    pass


class Busy(RuntimeError):
    pass


class InferenceService:
    def __init__(self, settings):
        self.settings = settings
        self.executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='inference')
        self.gate = asyncio.Semaphore(1)
        self.ready = False
        self.models = {}
        self.detectors = {}

    def _load(self):
        import torch
        from ..coordinate_inference import CoordinatePredictor
        from ..coordinate_detector import build_detector
        from ..npy_landmarks import sha256_file
        torch.set_num_threads(self.settings.torch_threads)
        bundle = self.settings.bundle_dir
        self.bundle = json.loads((bundle/'bundle.json').read_text())
        if set(self.bundle['models']) != {'sibi','bisindo'}:
            raise ValueError('Invalid service bundle languages')
        root = Path(__file__).resolve().parents[3]
        for language, entry in self.bundle['models'].items():
            directory = bundle/language
            if not {'selection.json','dataset.json','cnn_best.pt'}.issubset(entry['files']):
                raise ValueError('Missing required model bundle files')
            for name,digest in entry['files'].items():
                if Path(name).name != name or sha256_file(directory/name) != digest:
                    raise ValueError('Model bundle file hash mismatch')
            predictor = CoordinatePredictor(directory)
            if predictor.selection['language'] != language or predictor.svm is not None:
                raise ValueError('Unexpected service model')
            dataset = json.loads((directory/'dataset.json').read_text())
            detector = build_detector(dataset,predictor.selection,root,directory)
            self.models[language] = predictor
            self.detectors[language] = detector
        self.ready = True

    async def start(self):
        try:
            await asyncio.get_running_loop().run_in_executor(self.executor,self._load)
        except BaseException:
            await self.close()
            raise

    def descriptions(self):
        return {'bundle_version':self.bundle['bundle_version'],'models':{
            language:{'model_id':self.bundle['models'][language]['source_run'].split('/')[-1],
                'labels':model.classes,'recognition':'static_pose',
                'preprocessing':model.selection.get('runtime_preprocessing_sha256','baseline'),
                'limitations':(['J/Z are static poses, not movement recognition.'] if language=='sibi' else
                               ['D/F/P may be confused in webcam use.'])+['Scores are uncalibrated; no unknown/background class.']}
            for language,model in self.models.items()}}

    def _infer(self,language,payload,include_landmarks,mirrored):
        started = time.perf_counter()
        try:
            with Image.open(io.BytesIO(payload)) as image:
                if image.format not in {'JPEG','PNG'} or getattr(image,'is_animated',False):
                    raise BadFrame('Send a single JPEG or PNG image')
                if image.width*image.height > self.settings.max_frame_pixels:
                    raise BadFrame('Image dimensions exceed MAX_FRAME_PIXELS; resize webcam frames first')
                rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
        except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as exc:
            raise BadFrame('Invalid JPEG or PNG image') from exc
        if mirrored: rgb = np.ascontiguousarray(rgb[:,::-1])
        detector = self.detectors[language]
        detection_start = time.perf_counter()
        result = detector.extract(rgb,language,input_mirrored=False)
        detector_ms = (time.perf_counter()-detection_start)*1000
        prediction = None; classifier_start=time.perf_counter()
        if result.valid:
            prediction = self.models[language].predict(result.raw,result.height/result.width)
        classifier_ms = (time.perf_counter()-classifier_start)*1000 if prediction else 0.
        details = getattr(detector,'last_details',None) or {'method':'baseline','attempts':0,'recovered':False}
        response = dict(language=language,model_id=self.bundle['models'][language]['source_run'].split('/')[-1],
            status='predicted' if prediction else result.status,detection_status=result.status,
            label=prediction['label'] if prediction else None,score=prediction['score'] if prediction else None,
            score_type=prediction['score_type'] if prediction else None,hand_count=len(result.raw),
            width=result.width,height=result.height,
            preprocessing=dict(method=details['method'],attempts=details['attempts'],recovered=details['recovered'],
                               input_was_mirrored=mirrored,coordinates='unmirrored_normalized_xyz'),
            timings_ms=dict(detector=round(detector_ms,2),classifier=round(classifier_ms,2),total=round((time.perf_counter()-started)*1000,2)))
        response['landmarks'] = result.raw.tolist() if include_landmarks and np.isfinite(result.raw).all() else None
        return response

    async def predict(self,language,payload,include_landmarks=False,mirrored=False):
        if not self.ready: raise Busy('Service is not ready')
        try:
            await asyncio.wait_for(self.gate.acquire(),timeout=self.settings.queue_timeout_seconds)
        except TimeoutError as exc:
            raise Busy('Inference lane is busy; retry after the current frame') from exc
        future = asyncio.get_running_loop().run_in_executor(self.executor,self._infer,language,payload,include_landmarks,mirrored)
        # Cancellation must not unlock a graph while the executor is still using it.
        future.add_done_callback(lambda _:self.gate.release())
        return await asyncio.shield(future)

    def _close(self):
        self.ready=False
        for detector in self.detectors.values(): detector.close()
        self.detectors.clear()

    async def close(self):
        await asyncio.get_running_loop().run_in_executor(self.executor,self._close)
        self.executor.shutdown(wait=True)
