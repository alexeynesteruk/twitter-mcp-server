import os

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

HOST = os.getenv("APP_HOST") or "127.0.0.1"
PORT = os.getenv("APP_PORT")
