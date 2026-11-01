from pathlib import Path

from database.database import init_db
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_layer.monitoring.logger import setup_logging
from api_layer.routes import chat, conversations, history, home, upload

load_dotenv()

setup_logging()

app = FastAPI(title="Chatty gpt", description="An Agentic chatbot")


Path("uploads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

init_db()

#CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

#Routers
app.include_router(home.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(history.router)
app.include_router(upload.router)