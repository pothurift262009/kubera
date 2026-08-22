from flask import Flask, request, redirect
from kiteconnect import KiteConnect
import webbrowser

# ===== YOUR KEYS =====
API_KEY = "placeholder"
API_SECRET = "placeholder"

app = Flask(__name__)
kite = KiteConnect(api_key=API_KEY)

# ===== STEP 1: OPEN LOGIN =====
@app.route("/")
def login():
    login_url = kite.login_url()
    return redirect(login_url)

# ===== STEP 2: HANDLE REDIRECT =====
@app.route("/callback")
def callback():
    request_token = request.args.get("request_token")

    data = kite.generate_session(request_token, api_secret=API_SECRET)
    access_token = data["access_token"]

    kite.set_access_token(access_token)

    print("\n=== ACCESS TOKEN ===")
    print(access_token)

    return "Login successful. You can close this tab."

# ===== RUN SERVER =====
if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(port=5000)
