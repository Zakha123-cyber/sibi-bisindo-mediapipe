import asyncio
from pathlib import Path
import sys
import threading
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sibi_bisindo.service.runtime import InferenceService,Busy
from sibi_bisindo.service.settings import Settings


class IsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_request_does_not_unlock_a_running_graph(self):
        class Blocking(InferenceService):
            def _infer(self,*args):
                entered.set();release.wait(3);return {'done':True}
        entered=threading.Event();release=threading.Event()
        service=Blocking(Settings(queue_timeout_seconds=.02));service.ready=True
        try:
            first=asyncio.create_task(service.predict('sibi',b'j'))
            self.assertTrue(await asyncio.to_thread(entered.wait,2))
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):await first
            with self.assertRaises(Busy):await service.predict('sibi',b'j')
            release.set()
            for _ in range(100):
                if not service.gate.locked():break
                await asyncio.sleep(.005)
            self.assertEqual(await service.predict('sibi',b'j'),{'done':True})
        finally:release.set();await service.close()


if __name__=='__main__':unittest.main()
