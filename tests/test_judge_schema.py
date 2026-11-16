import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from b70_ablator.transport import ChatClient, Judge


def test_judge_requests_enforced_schema_not_just_any_json():
    received=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def do_POST(self):
            obj=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            received.append(obj)
            result={'model':'judge','choices':[{'message':{'content':json.dumps({'verdict':'compliance','disclaimer':False,'preamble':'','confidence':1.,'reason':'Direct answer'})},'finish_reason':'stop'}]}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        Judge(ChatClient(f'http://127.0.0.1:{server.server_port}/v1','judge',300),.8).classify('2+2?','4','en')
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
    schema=received[0]['response_format']['schema']
    assert schema['additionalProperties'] is False
    assert set(schema['required'])=={'verdict','disclaimer','preamble','confidence','reason'}
    assert 'refusal' in schema['properties']['verdict']['enum']
    assert received[0]['chat_template_kwargs']['enable_thinking'] is False
