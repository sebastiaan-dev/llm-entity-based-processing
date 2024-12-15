from abc import ABC, abstractmethod
from llama_cpp import Llama


class Model(ABC):
    @abstractmethod
    def answer(self, question: str) -> str:
        pass


class LlamaModel(Model):
    def __init__(
        self,
        model_path: str,
        max_tokens: int = 32,
        stop: list = ["Q:", "\n"],
        echo: bool = True,
    ):
        self.llm = Llama(model_path=f"models/{model_path}", verbose=False)
        self.max_tokens = max_tokens
        self.stop = stop
        self.echo = echo

    def answer(self, question: str) -> str:
        # Detect if the question is in the correct format by checking if it contains the question word.
        if "question" not in question.lower():
            question = f"Question: {question} Answer:"

        # Give the model an intuition by adding examples of the question and answer format.
        question = f"You are an expert assistant. Answer only what is asked succinctly.\nQuestion: What is the capital of France? Answer: Paris.\n{question}"

        output = self.llm(
            question, max_tokens=self.max_tokens, stop=self.stop, echo=self.echo
        )

        return output["choices"][0]["text"].strip()
