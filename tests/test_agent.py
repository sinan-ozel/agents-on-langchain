# pylint: disable=missing-module-docstring
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
# pylint: disable=wrong-import-order
import pytest
import time
from typing import Iterable

import numpy as np

from agents_on_langchain.base_agent import BaseAgent


class MockLLM:
    """Mock language model to simulate LLM behavior in tests."""

    def bind(self, skip_prompt=False):
        """Simulate binding to the model (no-op in this mock)."""
        return self

    def stream(self, prompt):
        """Simulate streaming output for the given prompt."""
        yield "Received:"
        for word in prompt.split(' '):
            yield word
        yield "Responding With:"
        for i in range(3):
            yield f"chunk{i + 1}"


class TestAgent(BaseAgent):
    """Test agent implementation with a mocked base_llm."""

    base_llm = MockLLM()
    q_and_a = [
        ("", "Received:TestpromptforResponding With:chunk1chunk2chunk3"),
        ("What is your name?", "TestAgent"),
    ]

    def __init__(self):
        """Initialize the TestAgent with nap time for simulated execution."""
        super().__init__()
        self.nap_time = 1  # Sleep time for the agent during the loop

    @property
    def version(self) -> str:
        """Return the agent version."""
        return "0.1.0"

    def listen(self, context: str) -> bool:
        """Accept all contexts for this test."""
        return True

    def _retrieve(self, q: str) -> list:
        """Return a mock retrieval result."""
        return [("Test document", {"metadata": "some info"})]

    def _prompt(self, q: str) -> str:
        """Generate a mock prompt based on the query."""
        return f"Test prompt for {q}"

    def respond(self, q: str) -> Iterable[str]:
        """Generate a response by streaming from the mock LLM."""
        prompt = self._prompt(q)
        for chunk in self.base_llm.bind(skip_prompt=True).stream(prompt):
            yield chunk

    def run(self) -> None:
        """Run the agent (for simulation purposes)."""
        print("TestAgent is running.")
        self.respond("Hello World")


@pytest.fixture
def agent():
    """Provide a new instance of TestAgent for each test."""
    return TestAgent()


def test_agent_start_stop(capfd, agent):
    """Test the start and stop functionality of the agent."""
    agent.start()

    # Allow the agent to "run" for a moment (simulate its execution)
    time.sleep(2)  # The agent runs for at least 1 second due to the nap_time

    # Capture the output generated during the agent's run
    captured = capfd.readouterr()

    # Check that the agent printed "TestAgent is running."
    assert "TestAgent is running." in captured.out

    # Stop the agent and assert that it stops without errors
    agent.stop()
    assert not agent._is_running


def test_agent_respond(agent):
    """Test the response generation of the agent."""
    query = "Test query"
    expected_response = [
        "Received:",
        "Test",
        "prompt",
        "for",
        "Test",
        "query",
        "Responding With:",
        "chunk1",
        "chunk2",
        "chunk3",
    ]
    # Verify the response matches the expected output
    response = list(agent.respond(query))
    assert response == expected_response


def test_receive(agent):
    """Test the receive function by validating the output format."""
    query = "Test query"
    expected_response = \
        "Received:TestpromptforTestqueryResponding With:chunk1chunk2chunk3"
    # Verify the response matches the expected output
    response = agent.receive(agent.respond(query))
    assert response == expected_response


def test_evaluate(agent):
    """Test the evaluation method of the agent."""
    # Evaluate the agent's performance
    score, correct_responses = agent.evaluate()
    # Verify the score is a float and the confusion matrix is a numpy array
    assert isinstance(score, str)
    assert isinstance(correct_responses, np.ndarray)
    assert score == '0.500'
    assert len(correct_responses) == 2
    assert correct_responses[0] == 1


def test_tell(agent):
    """Test the tell method of the agent."""
    assert agent.tell(agent, "Test message")
