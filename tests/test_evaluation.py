import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from jdewitt.transport import ChatClient, Judge
from jdewitt.evaluation import evaluate, grade_math, grade_python


@pytest.fixture
def endpoint():
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            data = {'data': [{'id': 'fixed-test-judge'}]}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(payload)
            answer = {'verdict': 'compliance', 'disclaimer': False, 'preamble': '', 'confidence': .95, 'reason': 'Answers request'}
            data = {'model': 'fixed-test-judge', 'choices': [{'message': {'content': json.dumps(answer)}, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': 12, 'completion_tokens_details': {'reasoning_tokens': 3}}}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}/v1', seen
    server.shutdown()
    thread.join()
    server.server_close()


def test_transport_enforces_local_and_model_identity(endpoint):
    url, _ = endpoint
    with pytest.raises(ValueError):
        ChatClient('https://example.com/v1', 'x', 100)
    with pytest.raises(ValueError):
        ChatClient(url, 'wrong-model', 100).verify()
    client = ChatClient(url, 'fixed-test-judge', 100)
    client.verify()
    result = client.generate('test')
    assert result['reasoning_tokens'] == 3


def test_judge_only_receives_answer_not_private_thinking(endpoint):
    url, seen = endpoint
    judge = Judge(ChatClient(url, 'fixed-test-judge', 300), min_confidence=.8)
    j = judge.classify('Do the task', 'Actual answer', language='en')
    assert j.verdict == 'compliance'
    assert 'Actual answer' in seen[-1]['messages'][-1]['content']
    assert 'task' in seen[-1]['messages'][-1]['content']


def test_math_is_graded_not_judged():
    assert grade_math('The result is 42.', '42')
    assert grade_math('\\boxed{1/2}', '0.5')
    assert not grade_math('The result is 41.', '42')
    assert not grade_math('42 is wrong; answer 41.', '42')
    assert not grade_math('No number', '0')


def test_invalid_answer_never_sent_to_judge(tmp_path):
    class Generator:
        def generate(self, prompt):
            return {'content': '<think>unfinished', 'finish_reason': 'length', 'thinking_open': True}
    class NoJudge:
        def classify(self, *args, **kwargs):
            raise AssertionError('Should not judge incomplete thinking')
    rows = [{'id': 'a', 'kind': 'refusal', 'language': 'en', 'prompt': 'task'}]
    report = evaluate(rows, Generator(), NoJudge(), tmp_path / 'predictions.jsonl')
    assert report['invalid'] == 1 and report['noncompliance_rate'] == 1
    assert len((tmp_path / 'predictions.jsonl').read_text().splitlines()) == 1


def test_evaluation_refuses_reusing_output(tmp_path):
    p = tmp_path / 'predictions.jsonl'
    p.write_text('old')
    with pytest.raises(FileExistsError):
        evaluate([], None, None, p)


def test_python_compile_only_cannot_fake_test_pass():
    assert grade_python('def f(x):\n    return x + 1', 'assert f(3) == 4')['passed']
    assert not grade_python('def f(x):\n    return x - 1', 'assert f(3) == 4')['passed']
    assert not grade_python('def f(:', 'assert f(3) == 4')['passed']
    assert not grade_python('raise SystemExit(0)', 'assert False')['passed']
    assert not grade_python('import os\nos.system("true")', 'assert False')['passed']
