# Agentic AI Chatbot

An agentic AI chatbot backend built with FastAPI, LangGraph, Google Gemini, and SQLite. The application supports conversational memory, tool calling, document retrieval, web search, and streaming AI responses using Server-Sent Events (SSE).

## Features

- **Agentic AI:** LangGraph-based agent with Google Gemini and tool calling.
- **Conversation memory:** Persistent conversation history using SQLite and LangGraph checkpoints.
- **Long-term memory:** Store and retrieve user-related memories.
- **RAG:** Upload documents and retrieve relevant information using Chroma.
- **Web search:** Retrieve current information using Tavily.
- **Calculator:** Perform mathematical calculations through an agent tool.
- **Streaming responses:** Stream AI responses to clients using SSE.
- **Conversation management:** Create conversations and retrieve chat history.
- **API validation:** Request and response schemas using Pydantic.
- **Testing:** Unit, API, integration, end-to-end, and concurrency tests.

## Tech Stack

| Component | Technology |
|---|---|
| API | FastAPI |
| AI framework | LangChain |
| Agent orchestration | LangGraph |
| LLM | Google Gemini |
| Database | SQLite |
| ORM | SQLAlchemy |
| Vector database | Chroma |
| Embeddings | Google Generative AI |
| Web search | Tavily |
| Validation | Pydantic |
| Testing | Pytest |
| Package management | uv |

## Project Structure

```text
backend/
├── ai_layer/
│   ├── agent/
│   │   ├── agent.py
│   │   └── mock_agent.py
│   ├── prompts/
│   │   └── prompt.py
│   ├── tools/
│   │   └── tools.py
│   ├── rag/
│   │   └── rag.py
│   └── schemas/
│       └── context.py
├── api_layer/
│   ├── main.py
│   ├── monitoring/
│   │   └── logger.py
│   ├── routes/
│   │   ├── chat.py
│   │   ├── conversations.py
│   │   ├── history.py
│   │   ├── home.py
│   │   └── upload.py
│   └── schemas/
│       └── chat.py
├── database/
│   ├── database.py
│   ├── models.py
│   └── repositories/
│       ├── conversation_repository.py
│       ├── message_repository.py
│       └── memory_repository.py
├── tests/
│   ├── unit/
│   ├── api/
│   ├── integration/
│   └── e2e/
│       └── concurrency/
├── data/
├── chroma_db/
├── uploads/
├── pyproject.toml
└── README.md
```

## Architecture

The application separates AI processing, API handling, and database operations.

- **AI layer:** Handles agent orchestration, prompts, tools, document retrieval, and model interactions.
- **API layer:** Exposes FastAPI endpoints, validates requests, manages streaming responses, and coordinates application operations.
- **Database layer:** Uses SQLAlchemy repositories to persist conversations, messages, and memories.

The agent uses LangGraph to manage its execution flow, including tool calls and conversation state.

## Requirements

- Python 3.12+
- uv
- Google Gemini API key
- Tavily API key

## Installation

Clone the repository and navigate to the backend directory:

```bash
git clone <repository-url>
cd agentic-chatbot/backend
```

Install dependencies:

```bash
uv sync
```

Create a `.env` file in the backend directory:

```env
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Add any additional environment variables required by your configuration.

## Running the Application

Start the FastAPI development server:

```bash
uv run uvicorn api_layer.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## API Functionality

The application provides endpoints for:

| Functionality | Description |
|---|---|
| Chat | Send messages and receive streamed AI responses |
| Conversations | Create and manage conversations |
| History | Retrieve persisted conversation messages |
| Document upload | Upload documents for retrieval |
| Home | Application entry endpoint |

Chat responses are streamed using Server-Sent Events (SSE).

The chat endpoint accepts a conversation identifier to maintain conversation continuity. When a new conversation is created, the application generates a unique identifier.

Refer to the FastAPI documentation for the exact request and response schemas.

## Testing

Run the complete test suite:

```bash
uv run pytest -v
```

Run a specific test category:

```bash
uv run pytest tests/unit -v
uv run pytest tests/api -v
uv run pytest tests/integration -v
uv run pytest tests/e2e -v
```

Concurrency tests are located in:

```text
tests/e2e/concurrency/
```

## Load Testing

Locust is used to simulate concurrent users and measure API performance.

Install the development dependencies:

```bash
uv sync
```

Start the application:

```bash
uv run uvicorn api_layer.main:app --reload
```

In another terminal, start Locust:

```bash
uv run locust -f tests/load/locustfile.py --host http://127.0.0.1:8000
```

Open the Locust dashboard:

```text
http://localhost:8089
```

Load testing with the real Gemini API is subject to the provider's rate limits and usage quotas.

## Current Scope

This project focuses on the backend architecture and AI application workflow.

Authentication, user-level authorization, production deployment, and comprehensive load testing remain separate areas of development.

The application should not be considered production-ready until security, access control, deployment, and performance requirements have been validated.

## License

Specify the license for this project.
