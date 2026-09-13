import os
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from requests_oauthlib import OAuth2Session
from pathlib import Path
from dotenv import load_dotenv

# The same .env systemd hands both services, found from this file rather than
# the working directory, so running it from anywhere reads the real one.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# No fallback values. A credential written into a default is a credential in
# the source, one careless `git add .` from being public.
CLIENT_ID = os.environ.get('OURA_CLIENT_ID')
CLIENT_SECRET = os.environ.get('OURA_CLIENT_SECRET')

AUTH_URL = 'https://cloud.ouraring.com/oauth/authorize'
TOKEN_URL = 'https://api.ouraring.com/oauth/token'
REDIRECT_URI = 'http://localhost:8080'
TOKEN_FILE = 'data/oura_token.json'

auth_code = None

class OAuthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global auth_code
        parsed_path = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed_path.query)
        
        if 'code' in query:
            auth_code = query['code'][0]
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Authorization successful! You can close this window.")
        else:
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"Authorization failed or no code provided.")
            
    def log_message(self, format, *args):
        pass  # Suppress HTTP server logs for clean terminal output

def main():
    if not CLIENT_ID or not CLIENT_SECRET:
        print("Please set OURA_CLIENT_ID and OURA_CLIENT_SECRET in your .env file or environment.")
        return
        
    oura = OAuth2Session(CLIENT_ID, redirect_uri=REDIRECT_URI)
    authorization_url, state = oura.authorization_url(AUTH_URL)
    
    print("Please go to the following URL and authorize access:")
    print(authorization_url)
    print("\nWaiting for callback on http://localhost:8080...")
    
    server = HTTPServer(('localhost', 8080), OAuthHandler)
    while not auth_code:
        server.handle_request()
        
    print("\nReceived auth code. Exchanging for token...")
    
    token = oura.fetch_token(
        TOKEN_URL,
        code=auth_code,
        client_secret=CLIENT_SECRET
    )
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(TOKEN_FILE) or '.', exist_ok=True)
    
    with open(TOKEN_FILE, 'w') as f:
        json.dump(token, f)
        
    print(f"Tokens successfully saved to {TOKEN_FILE}")

if __name__ == '__main__':
    main()

