import pytest

from agents_on_langchain.system_status import SystemStatusAgent


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


@pytest.fixture
def mock_llm():
    return MockLLM()


# Test for initializing and collecting system status with GPU
def test_system_status_with_gpu(mock_llm, mocker):
    # Mock torch.cuda methods to simulate a GPU being available
    mocker.patch('torch.cuda.is_available',
                 return_value=True)
    mocker.patch('torch.cuda.get_device_properties',
                 return_value=mocker.MagicMock(total_memory=16000000000))  # 16 GB
    mocker.patch('torch.cuda.get_device_name',
                 return_value="NVIDIA Tesla V100")

    agent = SystemStatusAgent(mock_llm)

    # Collect system status
    agent._collect_status()

    expected_gpu_memory = 16000000000 // (1024 * 1024)  # Convert to MB
    assert agent.gpu_memory == expected_gpu_memory  # 16 GB in MB
    assert agent.gpu_name == "NVIDIA Tesla V100"
    assert agent.model_name == "MockLLM"
    assert agent.status_collected is True


# Test for initializing and collecting system status without GPU
def test_system_status_without_gpu(mock_llm, mocker):
    # Mock torch.cuda methods to simulate no GPU being available
    mocker.patch('torch.cuda.is_available',
                 return_value=False)

    agent = SystemStatusAgent(mock_llm)

    # Collect system status
    agent._collect_status()

    assert agent.gpu_memory == "No GPU available"
    assert agent.gpu_name == "No GPU detected"
    assert agent.model_name == "MockLLM"
    assert agent.status_collected is True


# Test for _prompt when status is collected
def test_prompt_with_collected_status(mock_llm, mocker):
    mocker.patch('torch.cuda.is_available',
                 return_value=True)
    mocker.patch('torch.cuda.get_device_properties',
                 return_value=mocker.MagicMock(total_memory=16000000000))  # 16 GB
    mocker.patch('torch.cuda.get_device_name',
                 return_value="NVIDIA Tesla V100")

    agent = SystemStatusAgent(mock_llm)
    agent._collect_status()

    query = "What is the current system status?"
    prompt = agent._prompt(query)

    assert "System Information:" in prompt
    assert "GPU Name: NVIDIA Tesla V100" in prompt
    assert "GPU Memory: 15258 MB" in prompt
    assert "Model Name: MockLLM" in prompt
    assert "Query: What is the current system status?" in prompt


# Test for _prompt when status is not collected
def test_prompt_without_collected_status(mock_llm):
    agent = SystemStatusAgent(mock_llm)

    query = "What is the current system status?"
    prompt = agent._prompt(query)

    assert "System status has not been collected yet." in prompt


# Test for respond method with system status collected
def test_respond_with_collected_status(mock_llm, mocker):
    mocker.patch('torch.cuda.is_available',
                 return_value=True)
    mocker.patch('torch.cuda.get_device_properties',
                 return_value=mocker.MagicMock(total_memory=16000000000))  # 16 GB
    mocker.patch('torch.cuda.get_device_name',
                 return_value="NVIDIA Tesla V100")

    agent = SystemStatusAgent(mock_llm)
    agent._collect_status()

    query = "What is the current system status?"
    response = list(agent.respond(query))

    assert "Received:" in response[0]
    assert "chunk3" == response[-1]


# Test for the agent's run method (full cycle)
def test_agent_run(mock_llm, mocker):
    mocker.patch('torch.cuda.is_available',
                 return_value=True)
    mocker.patch('torch.cuda.get_device_properties',
                 return_value=mocker.MagicMock(total_memory=16000000000))  # 16 GB
    mocker.patch('torch.cuda.get_device_name',
                 return_value="NVIDIA Tesla V100")

    agent = SystemStatusAgent(mock_llm)

    # We simulate running the agent and collecting system status
    agent.run()

    expected_gpu_memory = 16000000000 // (1024 * 1024)  # Convert to MB
    assert agent.gpu_memory == expected_gpu_memory  # 16 GB in MB
    assert agent.gpu_name == "NVIDIA Tesla V100"
    assert agent.model_name == "MockLLM"
    assert agent.status_collected is True
