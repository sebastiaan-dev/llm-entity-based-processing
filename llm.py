import logging
import os

from llama_cpp import Llama

from debug import print_info
from models import Answer, Question


class LLM:
    def __init__(
        self,
        model_path: str,
        max_tokens: int = 64,
        stop: list = ["Q:", "\n"],
        echo: bool = True,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.llm = Llama(model_path, verbose=False)
        self.max_tokens = max_tokens
        self.stop = stop
        self.echo = echo

        self.logger = logging.getLogger(__name__)

    def answer(self, question: Question) -> Answer:
        """
        Generate an answer to the given question with the LLM model.

        The result is stripped of leading and trailing whitespaces.
        """

        output = self.llm(
            question.text, max_tokens=self.max_tokens, stop=self.stop, echo=self.echo
        )

        answer = Answer(
            question.id, output["choices"][0]["text"].strip().replace("\n", " ")
        )

        print_info("Question:", question.text)
        print_info("Answer:", answer.text)
        self.logger.info(f"question: {question.text}")
        self.logger.info(f"answer: {answer.text}")

        return answer
