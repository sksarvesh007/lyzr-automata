from typing import Any, Dict, List
import requests

from lyzr_automata.ai_models.model_base import AIModel
from lyzr_automata.data_models import FileResponse
from lyzr_automata.utils.resource_handler import ResourceBox

class ClaudeModel(AIModel):
    def __init__(self, api_key, anthropic_version, parameters: Dict[str, Any]):
        self.parameters = parameters
        # No OpenAI client initialization needed here
        self.api_key = api_key
        # Set the base URL for Anthropic's API
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.anthropic_version = anthropic_version

    def generate_text(
        self,
        task_id: str = None,
        system_persona: str = None,
        prompt: str = None,
        messages: List[dict] = None,
    ):
        # Prepare the messages for the API request
        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        # The data payload for the POST request
        data_payload = self.parameters
        data_payload["messages"] = messages
        
        if system_persona:
            data_payload["system"] = system_persona

        # Headers for the API request
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json"
        }

        response = requests.post(self.base_url, json=data_payload, headers=headers)

        if response.status_code == 200:
            response_data = response.json()
            if "content" in response_data and response_data["content"]:
                return response_data["content"][0]["text"]
            else:
                raise ValueError("No response content found.")
        else:
            raise Exception(f"An error occurred: {response.status_code}, {response.text}")
        
    def generate_image(
        self, task_id: str, prompt: str, resource_box: ResourceBox
    ) -> FileResponse:
        # kept empty for future use
        pass