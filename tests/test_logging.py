# pylint: disable=missing-module-docstring
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
# pylint: disable=wrong-import-order
import pytest
import logging
from typing import Iterable, List, Tuple, Callable
import numpy as np

from agents_on_langchain.base_agent import BaseAgent


class TestAgent(BaseAgent):
    """
    A simple implementation of BaseAgent for testing purposes.
    This class simulates a basic agent that can listen, retrieve, and respond to queries
    for the purpose of testing agent functionalities.
    """

    @property
    def version(self) -> str:
        """
        Returns the version of the agent.

        Returns:
            str: The version of the agent.
        """
        return "1.0"

    def listen(self, context: str) -> bool:
        """
        Simulates the agent listening to a given context.

        Args:
            context (str): The context to listen to.

        Returns:
            bool: Always returns True for this basic implementation.
        """
        return True

    def _retrieve(self, q: str) -> List[Tuple[str, dict]]:
        """
        Simulates retrieving relevant data based on the given query.

        Args:
            q (str): The query string.

        Returns:
            List[Tuple[str, dict]]: A list containing the query paired with an
            empty dictionary.
        """
        return [(q, {})]

    def _prompt(self, q: str) -> str:
        """
        Simulates generating a prompt based on the given query.

        Args:
            q (str): The query string.

        Returns:
            str: A formatted prompt string.
        """
        return f"Prompted: {q}"

    def respond(self, q: str) -> Iterable[str]:
        """
        Simulates generating a response to a given query.

        Args:
            q (str): The query string.

        Yields:
            str: A simulated response string.
        """
        yield f"Response to: {q}"

    def run(self) -> None:
        """
        Placeholder method for running the agent. Does nothing in this test
        implementation.
        """


@pytest.fixture
def agent():
    """
    Fixture that initializes a TestAgent instance for testing.

    Returns:
        TestAgent: A new instance of TestAgent.
    """
    return TestAgent()


@pytest.fixture
def other_agent():
    """
    Fixture that initializes another TestAgent instance for testing.

    Returns:
        TestAgent: A new instance of TestAgent.
    """
    return TestAgent()


@pytest.fixture
def caplog_debug(caplog):
    """
    Fixture to set the logging level to DEBUG for capturing log messages.

    Args:
        caplog: The pytest fixture for capturing log messages.

    Returns:
        caplog: The caplog fixture with the logging level set to DEBUG.
    """
    caplog.set_level(logging.DEBUG)
    return caplog


def test_logger_initialization(agent):
    """
    Test that the agent's logger is initialized correctly with the expected name
    and level.

    Args:
        agent (TestAgent): The TestAgent instance to test.
    """
    assert agent.logger.name == "TestAgent"
    assert agent.logger.level == logging.WARNING


def test_logging_ask(agent, other_agent, caplog_debug):
    """
    Test that calling ask() logs the expected debug messages.

    Args:
        agent (TestAgent): The agent instance.
        other_agent (TestAgent): Another agent instance to pass as a parameter to `ask`.
        caplog_debug: The fixture to capture logs.
    """
    agent.logger.level = logging.DEBUG
    q = "What is AI?"
    list(agent.ask(other_agent, q))  # Trigger the logging
    print(caplog_debug.text.split("\n"))

    assert any("Asking" in message and q in message
               for message in caplog_debug.text.split("\n"))
    assert any("Received" in message and q in message
               for message in caplog_debug.text.split("\n"))


def test_logging_tell(agent, other_agent, caplog_debug):
    """
    Test that calling tell() logs the expected debug messages.

    Args:
        agent (TestAgent): The agent instance.
        other_agent (TestAgent): Another agent instance.
        caplog_debug: The fixture to capture logs.
    """
    agent.logger.level = logging.DEBUG
    context = "New context"
    agent.tell(other_agent, context)

    assert any("Telling" in message and context in message
               for message in caplog_debug.text.split("\n"))


def test_evaluate(agent):
    """
    Test the evaluate method to ensure accuracy calculation works as expected.

    Args:
        agent (TestAgent): The agent instance to test.
    """
    agent.q_and_a = [("Test?", "Response to: Test?")]
    accuracy, matches = agent.evaluate()
    assert accuracy == "1.000"
    assert matches.tolist() == [1]


def test_start_stop(agent, caplog_debug):
    """
    Test that start() and stop() log the expected info messages.

    Args:
        agent (TestAgent): The agent instance.
        caplog_debug: The fixture to capture logs.
    """
    agent.logger.level = logging.INFO
    agent.start()
    assert any("Starting agent" in message for message in caplog_debug.text.split("\n"))

    agent.stop()
    assert any("Stopping agent" in message for message in caplog_debug.text.split("\n"))
