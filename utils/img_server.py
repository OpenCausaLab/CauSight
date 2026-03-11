import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from pathlib import Path
from urllib.parse import quote, unquote
import mimetypes
import time

class FlexibleImageHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        try:
            file_path = unquote(self.path[1:])
            
            if not os.path.isabs(file_path):
                file_path = '/' + file_path
            
            if not os.path.exists(file_path):
                self.send_error(404, f"File not found: {file_path}")
                return
            
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type is None:
                content_type = 'application/octet-stream'
            
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            with open(file_path, 'rb') as f:
                self.wfile.write(f.read())
                
        except Exception as e:
            self.send_error(500, f"Server error: {str(e)}")
            print(f"Server error while handling request: {e}")

class ImageServer:
    def __init__(self, port=18901):
        self.port = port
        self.server = None
        self.thread = None
        self._ready = threading.Event()
        
    def start(self):
        if self.server is not None:
            print("Server is already running")
            return
        
        def run_server():
            try:
                self.server = HTTPServer(('localhost', self.port), FlexibleImageHandler)
                print(f"Image server started on port: {self.port}")
                self._ready.set()
                self.server.serve_forever()
            except Exception as e:
                print(f"Server startup failed: {e}")
                self._ready.set()
        
        self.thread = threading.Thread(target=run_server)
        self.thread.daemon = True
        self._ready.clear()
        self.thread.start()
        
        self._ready.wait(timeout=5)
        if not self._ready.is_set():
            raise RuntimeError("Server startup timeout")
        
    def stop(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
            self.thread = None
            print("Server closed")
            self._ready.clear()
    
    def get_url(self, local_path):
        if not self._ready.is_set():
            raise RuntimeError("Server not started or startup failed")
        
        abs_path = os.path.abspath(local_path)
        encoded_path = quote(abs_path)
        return f"http://localhost:{self.port}/{encoded_path.lstrip('/')}"

def process_image_path(server, image_paths):
    if isinstance(image_paths, str):
        return server.get_url(image_paths)
    else:
        return [server.get_url(path) for path in image_paths]
