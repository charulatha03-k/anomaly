import http.server
import socketserver

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

if __name__ == '__main__':
    port = 8000
    with socketserver.TCPServer(("", port), CORSRequestHandler) as httpd:
        print(f"Serving with CORS on port {port}")
        httpd.serve_forever()
