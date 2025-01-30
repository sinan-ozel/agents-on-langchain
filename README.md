A lightweight framework to create agents based on the LangChain `BaseModel` interface.

The idea is to make it easy to a multi-agent platform.

# Usage

## Basic Example
(TODO)

## Complex Journal Multi-Agent Platform
(TODO)

## 

# Philosophy

I went with three guiding principles in writing this model.

1. Everything is an agent: tools, buses, orchestrators are all agents.
2. Final code should show the flow: You should be able scale while being able to
   see how agents connect to each other. This means that each agent relationship
   should be at most one line of code.
3. Agents are minimal building blocks: one prompt per agent, one vector store
   per agent, one model per agent.

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
