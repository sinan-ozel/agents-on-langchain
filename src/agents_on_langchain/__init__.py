"""A lightweight Python package for creating agents on Langchain"""


# Instructioons: Update the version according to semver. If the version is
# different from the previous versions, this will automatically create a
# "stable" release on PyPI. Otherwise, all merges to the main branch will create
# a "pre-release" on PyPI. In that case, the version will be suffixed with
# ".dev" and date and the git commit hash

__version__ = "0.1.1"


def streamed_response(func):
    """
    Wrapper function for the respond method to handle streaming responses.
    """
    def new_func(self, *args, **kwargs):
        prompt = func(self, *args, **kwargs)
        for chunk in self.base_llm.bind(skip_prompt=True).stream(prompt):
            yield chunk
    return new_func
