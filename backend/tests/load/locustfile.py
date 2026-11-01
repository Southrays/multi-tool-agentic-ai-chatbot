import json
from uuid import uuid4

from locust import HttpUser, between, task


class ChatbotUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        # Each simulated user gets an independent conversation.
        self.thread_id = str(uuid4())

    @task
    def send_chat_message(self):
        payload = {
            "message": "Explain what an AI agent is in simple terms.",
            "thread_id": self.thread_id,
            "model": "gemini-3.8-flash",
        }

        with self.client.post(
            "/chat/",
            json=payload,
            stream=True,
            timeout=120,
            catch_response=True,
            name="/chat/ (SSE)",
        ) as response:

            if response.status_code != 200:
                response.failure(
                    f"Unexpected status code: {response.status_code}"
                )
                return

            if "text/event-stream" not in response.headers.get(
                "Content-Type", ""
            ):
                response.failure("Response is not an SSE stream")
                return

            received_done = False

            try:
                for line in response.iter_lines():
                    if not line:
                        continue

                    line = line.decode("utf-8")

                    if not line.startswith("data: "):
                        continue

                    event_data = json.loads(line[6:])

                    if "error" in event_data:
                        response.failure(event_data["error"])
                        return

                    if event_data.get("done") is True:
                        received_done = True

                if not received_done:
                    response.failure("Stream ended without a done event")

            except (ValueError, json.JSONDecodeError) as error:
                response.failure(f"Invalid SSE event: {error}")
