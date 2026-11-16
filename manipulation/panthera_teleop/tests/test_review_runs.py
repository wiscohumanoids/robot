import importlib.util
import json
from pathlib import Path
import tempfile
import threading
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'scripts/review_runs.py'
spec = importlib.util.spec_from_file_location('review_runs', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        source = self.root / 'review'
        episode = source / 'run-1'
        episode.mkdir(parents=True)
        (episode / 'metadata.json').write_text(json.dumps(dict(start_monotonic_ns=1000000000, complete=True, label='review')))
        (episode / 'states.jsonl').write_text(json.dumps(dict(monotonic_ns=11000000000))+'\n')
        (episode / 'frames.jsonl').write_text(json.dumps(dict(monotonic_ns=3000000000, camera='overhead', path='images/overhead/0.jpg'))+'\n')
        template = self.root / 'template.json'
        template.write_text(json.dumps(dict(fps=20, episodes=[dict(id='old')], image_crops_xyxy={})))
        self.store = module.ReviewStore(source, self.root / 'annotations.json', template)

    def test_clock_and_round_trip(self):
        self.assertEqual(self.store.episode('run-1')['cameras']['overhead'][0]['t'], 2)
        self.assertIsNone(self.store.listing()['episodes'][0]['edit'])
        self.store.save('run-1', dict(start_s=2, end_s=9, reason='Recovery'))
        reopened = module.ReviewStore(self.store.source, self.store.annotations, self.store.template)
        self.assertTrue(reopened.listing()['episodes'][0]['edit']['review_approved'])
        self.assertEqual(reopened.spec()['fps'], 20)
        self.store.save('run-1', dict(start_s=3, end_s=8))
        self.assertEqual(len(self.store.spec()['episodes']), 1)
        self.assertEqual(self.store.spec()['episodes'][0]['start_s'], 3)
        self.assertEqual(json.loads((self.store.source / 'run-1/metadata.json').read_text())['label'], 'review')

    def test_http_save_and_reload(self):
        server = module.ThreadingHTTPServer(('127.0.0.1', 0), module.handler_for(self.store))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_port}'
            request = Request(url + '/api/run/run-1', data=json.dumps(dict(start_s=2, end_s=8)).encode(),
                              headers={'Content-Type': 'application/json'}, method='POST')
            with urlopen(request) as response:
                self.assertTrue(json.load(response)['review_approved'])
            with urlopen(url + '/api/runs') as response:
                self.assertEqual(json.load(response)['episodes'][0]['edit']['end_s'], 8)
            request.add_header('Origin', 'https://unrelated.example')
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, 403)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_notes_do_not_approve_or_change_crop(self):
        self.store.update_details('run-1', {'notes': 'Needs another look'})
        self.assertFalse(self.store.annotations.exists())
        self.assertIsNone(self.store.listing()['episodes'][0]['edit'])
        self.assertEqual(self.store.episode('run-1')['notes'], 'Needs another look')
        self.store.save('run-1', dict(start_s=2, end_s=8))
        before = self.store.spec()['episodes'][0]
        self.store.update_details('run-1', {'notes': ''})
        self.assertEqual(self.store.spec()['episodes'][0], before)
        self.assertEqual(self.store.episode('run-1')['notes'], '')

    def test_categories_preserve_access_and_exclude_discarded(self):
        self.store.save('run-1', dict(start_s=2, end_s=8))
        self.store.update_details('run-1', {'category': 'kept', 'notes': 'Good recovery'})
        self.assertFalse((self.store.source / 'run-1').exists())
        self.assertTrue((self.root / 'kept/run-1/frames.jsonl').exists())
        self.assertEqual(self.store.episode('run-1')['label'], 'kept')
        self.assertEqual(self.store.spec()['episodes'][0]['source_category'], 'kept')
        self.assertEqual(self.store.listing()['episodes'][0]['label'], 'kept')
        self.store.update_details('run-1', {'category': 'discarded'})
        self.assertEqual(self.store.spec()['episodes'], [])
        with self.assertRaises(ValueError):
            self.store.save('run-1', dict(start_s=2, end_s=8))
        self.store.update_details('run-1', {'category': 'review'})
        self.assertEqual(self.store.episode('run-1')['notes'], 'Good recovery')
        self.assertIsNone(self.store.listing()['episodes'][0]['edit'])
        with self.assertRaises(ValueError):
            self.store.update_details('run-1', {'category': '../elsewhere'})

    def test_listing_includes_unedited_kept_runs(self):
        import shutil
        kept = self.root / 'kept' / 'run-2'
        shutil.copytree(self.store.source / 'run-1', kept)
        metadata = json.loads((kept / 'metadata.json').read_text())
        metadata['label'] = 'kept'
        (kept / 'metadata.json').write_text(json.dumps(metadata))
        rows = {r['id']: r for r in self.store.listing()['episodes']}
        self.assertEqual(rows['run-2']['label'], 'kept')
        self.assertIsNone(rows['run-2']['edit'])
        self.assertTrue((kept / 'frames.jsonl').exists())

    def test_invalid_intervals_and_paths(self):
        for start, end in [(-1, 5), (5, 5), (5, 11), (float('nan'), 8), (0, float('inf')), (2, 2.1)]:
            with self.assertRaises(ValueError):
                self.store.save('run-1', dict(start_s=start, end_s=end))
        with self.assertRaises(ValueError):
            self.store.episode('../template.json')
        self.assertFalse(self.store.annotations.exists())


if __name__ == '__main__':
    unittest.main()
