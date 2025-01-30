"""
SystemStatusAgent module for hardware and model status monitoring.

This module defines the SystemStatusAgent class, which collects and tracks
system information such as GPU availability, GPU model, and language model name.
It updates this information periodically and provides it as context for query responses.
"""
import torch
from typing import Iterable
from langchain_core.language_models.llms import BaseLLM
from .base_agent import BaseAgent


class SystemStatusAgent(BaseAgent):
    """
    An AI-powered agent that monitors system hardware and language model details.

    Features:
    - Collects GPU memory availability.
    - Retrieves the GPU model name.
    - Identifies the language model in use.
    - Stores this information for use in query responses.
    - Updates the information every 2 hours.

    Args:
        llm (BaseLLM): The language model used for responses.
    """
    version = "01"

    def __init__(self, llm: BaseLLM):
        """
        Initialize the SystemStatusAgent.

        Args:
            llm (BaseLLM): Language model for generating responses.
        """
        super(SystemStatusAgent, self).__init__()
        self.base_llm = llm
        self.nap_time = 7200  # 2 hours
        self.status_collected = False
        self.gpu_memory = None
        self.gpu_name = None
        self.model_name = None

    def listen(self, context: str) -> bool:
        """
        This agent does not process external context.
        """
        return False

    def _collect_status(self):
        """
        Collect system status including GPU details and model name.
        """
        if torch.cuda.is_available():
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)  # Convert to MB
            self.gpu_name = torch.cuda.get_device_name(0)
        else:
            self.gpu_memory = "No GPU available"
            self.gpu_name = "No GPU detected"

        self.model_name = self.base_llm.__class__.__name__
        self.status_collected = True

    def _prompt(self, q: str) -> str:
        """
        Generate a prompt using the collected system information.
        If status has not been collected yet, generate a default response.
        """
        if self.status_collected:
            return (f"System Information:\n"
                    f"GPU Name: {self.gpu_name}\n"
                    f"GPU Memory: {self.gpu_memory} MB\n"
                    f"Model Name: {self.model_name}\n\n"
                    f"Query: {q}")
        else:
            return "System status has not been collected yet. "\
                   "Please wait for the next update cycle."

    def respond(self, q: str) -> Iterable[str]:
        """
        Generate a response using system information as context.
        """
        prompt = self._prompt(q)
        for chunk in self.base_llm.bind(skip_prompt=True).stream(prompt):
            yield chunk

    def run(self) -> None:
        """
        Collect system information and sleep for 2 hours before updating again.
        """
        self._collect_status()
