from ai_layer.agent.agent import get_agent
from database.database import init_db
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

init_db()

agent = get_agent("gemini-3.8-flash")

config: RunnableConfig = {"configurable": {"thread_id": "test-thread-id"}}

for message_chunk, metadata in agent.stream(
    {"messages": [HumanMessage(content="How are you?")]},
    config=config,
    stream_mode="messages"
):
    if metadata.get("langgraph_node") == "chatbot" and message_chunk.content:
        content = message_chunk.content

        if isinstance(content, str):
            print(content, end="", flush=True)

        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    print(block["text"], end="", flush=True)
