A lightweight framework to create agents based on the LangChain `BaseModel` interface.

# Usage


# Philosophy

I went with three guiding principles in writing this model.

1. Everything is an agent: tools, buses, orchestrators are all agents.
2. Final code should show the flow: You should be able scale while being able to
   see how agents connect to each other. This means that each agent relationship
   should be at most one line of code.
3. Agents are minimal building blocks: one prompt per agent, one vector store
   per agent, one model per agent.

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
