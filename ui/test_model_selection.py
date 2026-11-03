"""Consumer model choices and provider routing, without model calls."""
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ui import server, test_artsy


class ModelSelectionTests(unittest.TestCase):
    def test_four_models_reach_the_correct_provider_for_drawings_and_edits(self):
        served = [{'id': model['id'], 'model_path': 'tinker://fixture/sampler',
                   'base_model': server.LARGE_MODEL_ID}
                  for model in server.ARTSY_MODELS if model['id'].startswith('artsy/')]
        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(server, 'bridge_model_catalog', return_value=served):
            root = Path(temporary)
            ident, _ = test_artsy.ArtsyTests().completed(root)
            with patch.object(server, 'JOBS', root):
                for index, model in enumerate(server.ARTSY_MODELS):
                    for parent in (None, ident):
                        with self.subTest(model=model['label'], parent=parent):
                            body = {'prompt': 'A blue fox', 'model': model['id']}
                            if parent:
                                body['parent_id'] = parent
                            request = server.artsy_request(body)
                            self.assertEqual(request['model'], model['id'])
                            self.assertEqual(request.get('parent_id'), parent)
                            self.assertEqual(request['max_turns'], 10)
                            folder = root / f'run-{index}-{parent or "new"}'
                            folder.mkdir()
                            with patch.object(server.SUPERVISOR, 'spawn') as run:
                                run.return_value.wait.return_value = 0
                                run.return_value.pid = 12345
                                server.run_job(folder, request)
                            environment = run.call_args.kwargs['env']
                            expected = (server.OPENROUTER_MODEL_URL if model.get('direct')
                                        else 'http://127.0.0.1:8080/v1')
                            self.assertEqual(environment['ARTSY_MODEL_URL'], expected)
                            self.assertIn(model['id'], run.call_args.args[1])

    def test_unknown_and_unavailable_models_are_rejected(self):
        for model in ('other/model', 'Qwen/Qwen3.5-4B', '', None, []):
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, 'Choose Base'):
                server.artsy_request({'prompt': 'fox', 'model': model})
        with patch.object(server, 'bridge_model_catalog', return_value=None):
            for model in server.ARTSY_MODELS[1:3]:
                with self.subTest(model=model), self.assertRaisesRegex(ValueError, 'unavailable'):
                    server.artsy_request({'prompt': 'fox', 'model': model['id']})

    def test_home_renders_exactly_four_models_with_luna_selected(self):
        class Options(HTMLParser):
            def __init__(self):
                super().__init__()
                self.options = []

            def handle_starttag(self, tag, attrs):
                if tag == 'option':
                    self.options.append(dict(attrs))

        handler = object.__new__(server.Handler)
        handler.headers = {'Host': f'127.0.0.1:{server.PORT}'}
        for route in ('/', '/index.html'):
            handler.path = route
            with patch.object(handler, 'send_page') as page:
                handler.do_GET()
            parser = Options()
            parser.feed(page.call_args.args[0])
            self.assertEqual([option['value'] for option in parser.options],
                             [model['id'] for model in server.ARTSY_MODELS])
            self.assertEqual([option['value'] for option in parser.options if 'selected' in option],
                             ['openai/gpt-6-luna'])


if __name__ == '__main__':
    unittest.main()
