"""
SummaryAgent module for AI-driven summarization.

This module defines the SummaryAgent class, which processes and summarizes
textual information using a language model. It maintains a rolling context,
automatically condensing information when necessary.
"""
from collections import deque
from typing import Iterable, List, Tuple

from langchain_core.language_models.llms import BaseLLM

from .base_agent import BaseAgent


class SummaryAgent(BaseAgent):
    """
    An AI-powered agent that listens to contextual information,
    processes it, and generates summaries using a language model.

    Features:
    - Maintains a FIFO queue for storing recent context.
    - Summarizes content when it exceeds a predefined word limit.
    - Condenses multiple entries into a single summary when necessary.
    - Generates context-aware responses to queries.

    Args:
        llm (BaseLLM): The language model used for text summarization.
        max_context (int): Maximum number of context entries to store.
        summary_length (int): Target word count for condensed summaries.
        detailed_summary_length (int): Target word count for full summaries.
    """
    version = '01'

    def __init__(self, llm: BaseLLM):
        """
        Initialize the SummaryAgent.

        Args:
            llm (BaseLLM): Language model for generating summaries.
            max_context (int): Maximum length of the FIFO context array.
            summary_length (int): Target word count for standard summaries.
            detailed_summary_length (int): Target word count for detailed summaries.
        """
        super(SummaryAgent, self).__init__()
        self.base_llm = llm
        self.context_queue = deque(maxlen=10)  # FIFO queue
        self.summary_length = 100
        self.detailed_summary_length = 250
        self.nap_time = 5

    def listen(self, context: str) -> bool:
        """
        Accepts contextual information and stores it in a FIFO queue.
        """
        self.context_queue.append(context)
        return True

    def _retrieve(self, q: str) -> List[Tuple[str, dict]]:
        """
        Retrieves relevant information for a given query and summarizes it.
        If q is empty, summarizes the current context.
        """
        prompt = self._prompt(q)
        summary = self.base_llm.generate(prompt)  # Using LLM to generate summary
        return [(summary, {"source": "context"})]

    def _prompt(self, q: str) -> str:
        """
        Generates a summarization prompt.
        """
        if self.context_queue:
            context_text = " ".join(self.context_queue)
        else:
            context_text = "No context available."

        if q.strip():
            return (f"Given the context:\n{context_text}\n\n"
                    f"and the query: '{q}', generate a "
                    f"{self.detailed_summary_length}-word summary.")
        else:
            return (f"Summarize the following text into "
                    f"{self.detailed_summary_length} words:\n{context_text}")

    def respond(self, q: str) -> Iterable[str]:
        """
        Generates a summary response for a given query.
        """
        prompt = self._prompt(q)
        response = self.base_llm.generate(prompt)
        yield response

    def run(self) -> None:
        """
        Runs the agent in a loop, processing the context queue every 5 seconds.
        """
        # Step 1: Summarize if total word count > 250
        context_text = " ".join(self.context_queue)
        if len(context_text.split()) > 250:
            prompt = self._prompt("")
            summary = self.base_llm.generate(prompt)
            self.context_queue.clear()
            self.context_queue.append(summary)

        # Step 2: Summarize last 7 entries if more than 5 exist
        if len(self.context_queue) > 5:
            last_entries = list(self.context_queue)[-7:]
            temp_context = " ".join(last_entries)
            prompt = f"Summarize the following text into "\
                     f"{self.summary_length} words:\n{temp_context}"
            summary = self.base_llm.generate(prompt)

            # Remove last 7, insert summary at index 4
            for _ in range(7):
                self.context_queue.pop()
            self.context_queue.insert(4, summary)
