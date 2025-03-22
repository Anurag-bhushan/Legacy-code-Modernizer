from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from code_modernizer import EnhancedAIModernizer
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the modernizer with API key
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
modernizer = EnhancedAIModernizer(GEMINI_API_KEY)

class ModernizeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/api/modernize':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))

            try:
                # Get code and language from request
                code = data.get('code', '')
                language = data.get('language', 'python')

                # Modernize the code
                result = modernizer.modernize_code(code, language)

                # Prepare response
                response = {
                    'success': result.success,
                    'modernized_code': result.modernized_code,
                    'changes_made': result.changes_made,
                    'language': result.language,
                    'frameworks_detected': result.frameworks_detected
                }

                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'error': str(e)
                }).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, ModernizeHandler)
    print(f'Starting server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()