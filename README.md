A lightweight framework to create "microagents" based on the LangChain `BaseModel` interface.

The idea is to make it easy to a multi-agent platform.

# Usage

## Basic Example
The following deploys an agent based on a self-hosted hugging face model as
an endpoint.

```python
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface.llms import HuggingFacePipeline
from agents_on_langchain.base_agent import BaseAgent

model = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3',
                                             torch_dtype=torch.bfloat16,
                                             device_map="auto")
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')


class RequestClassifierAgent(BaseAgent):
    version = '01'
    base_llm = HuggingFacePipeline(
        pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            temperature=0.1,
            max_new_tokens=64,
            do_sample=True
        )
    )

    q_and_a = [
        ("Tell me about the languages of Eberron.", "lookup"),
        ("War veteran", "character"),
        ("Create a House Cannith item.", "original"),
        ("Create the details of a town on the border between Zilargo and Breland.", "original"),
        ("Groggy comic relief", "character"),
        ("Tell me about fashion in the five nations.", "lookup"),
        ("Find for me a magic lipstick.", "lookup"),
    ]

    def listen(self, context: str) -> bool:
        """This agent does not listen to any contextual information"""
        return False

    def __init__(self):
        pass

    def _retrieve(self, q: str) -> List[Tuple[str, dict]]:
        return []

    def _prompt(self, q: str):
        return dedent(f"""[INST]
        User is making a request. Classify the request into one of the three categories. Response with only one word.
        If the user is asking to create some original content, respond with the word "original".
        If this is a request to create a character or NPC, or it looks like it is decribing a D&D character, respond with the word "character".
        If the user is making a request to find out information, respond with the word "lookup".

        Q:
        {q}

        A:
        [/INST]""")

    def respond(self, q: str) -> Iterable[str]:
        category = self.base_llm.bind(skip_prompt=True).invoke(self._prompt(q))

        yield category.strip()

    def run() -> None:
        pass


class Message(BaseModel):
    content: str


request_classifier = RequestClassifierAgent()
app = FastAPI()


@app.post("/classify")
def classify(message: Message):
    generator = supervisor.respond(message.content)
    return StreamingResponse(request_classifier, media_type="text/event-stream")


```
## Complex Exmaple
See here:
https://github.com/sinan-ozel/eberron-llm/tree/main/multi-agent-servers/eberron-agent-server/app

## 

# Philosophy

I went with three guiding principles in writing this model.

1. Everything is an agent: tools, buses, orchestrators are all agents.
2. Final code should show the flow: You should be able scale while being able to
   see how agents connect to each other. This means that each agent relationship
   should be at most one line of code.
3. Agents are minimal building blocks: one prompt per agent, one vector store
   per agent, one model per agent. In other words, they are _microagents_.

# Deployment
The mode of deployment is as follows:
Make sure that there is a model and an API, based on the `BaseModel`in LangChain.
Put all agents in the same piece of code.
Run this code on a loop - the loop can also be an "orchestrator" agent.

Note that you can also host the model on the same pod / instance / computer as 
the agents. This is how I (the author) tested it.


# Development

## Requirements

Just Docker. If you want to develop, you can use the .devcontainer on VS Code,
you don't need to install anything.

This works with VS Code, however, if you want to use another IDE, you can also
use the `Dockerfile.dev` to create your development environment.

## Testing

Run:

```
docker-compose run --rm --build test
```


## Contributing

1. Branch out
2. Add new code.
3. Add tests.
4. Push.
5. Make a pull request.
