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
            # 解码URL路径，得到绝对文件路径
            file_path = unquote(self.path[1:])  # 去掉开头的'/'
            
            # 确保使用绝对路径
            if not os.path.isabs(file_path):
                # 如果不是绝对路径，添加根目录
                file_path = '/' + file_path
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                self.send_error(404, f"File not found: {file_path}")
                return
            
            # 获取文件MIME类型
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type is None:
                content_type = 'application/octet-stream'
            
            # 发送文件
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
        """启动服务器 - 不需要指定目录"""
        if self.server is not None:
            print("服务器已经在运行中")
            return
        
        def run_server():
            try:
                self.server = HTTPServer(('localhost', self.port), FlexibleImageHandler)
                print(f"图片服务器已启动，端口: {self.port}")
                self._ready.set()
                self.server.serve_forever()
            except Exception as e:
                print(f"服务器启动失败: {e}")
                self._ready.set()
        
        self.thread = threading.Thread(target=run_server)
        self.thread.daemon = True
        self._ready.clear()
        self.thread.start()
        
        self._ready.wait(timeout=5)
        if not self._ready.is_set():
            raise RuntimeError("服务器启动超时")
        
    def stop(self):
        """停止服务器"""
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
            self.thread = None
            print("服务器已关闭")
            self._ready.clear()
    
    def get_url(self, local_path):
        if not self._ready.is_set():
            raise RuntimeError("服务器未启动或启动失败")
        
        # 获取绝对路径
        abs_path = os.path.abspath(local_path)
        encoded_path = quote(abs_path)
        return f"http://localhost:{self.port}/{encoded_path.lstrip('/')}"

def process_image_path(server, image_paths):
    if isinstance(image_paths, str):
        return server.get_url(image_paths)
    else:
        return [server.get_url(path) for path in image_paths]
