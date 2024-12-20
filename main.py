import logging
from logging.handlers import RotatingFileHandler

from colorama import init

from extractor import EntityExtractor
from file_manager import FileManager
from linker import EntityLinker
from llm import LLM
from models import Problem, Solution
from solver import Solver
from verifier import Verifier

file_handler = RotatingFileHandler(
    "logs/app.log", maxBytes=10 * 1024 * 1024, backupCount=10
)
logging.basicConfig(level=logging.INFO, handlers=[file_handler])


class Pipeline:
    def __init__(
        self,
        file_manager: FileManager,
        model: LLM,
        extractor: EntityExtractor,
        linker: EntityLinker,
        solver: Solver,
        verifier: Verifier,
    ):
        self.file_manager = file_manager
        self.model = model
        self.extractor = extractor
        self.linker = linker
        self.solver = solver
        self.verifier = verifier

        self.logger = logging.getLogger(__name__)

    def resolve(self):
        """
        This function acts as a pipeline for the different stages of the resolution process.

        1. Read the input data into a list of Question objects.
        2. Generate the answers for each question using an LLM.
        3. Extract entities from the questions and answers.
        4. Extract the formatted answer from the generated answer.
        5. Check the correctness of the extracted answer.
        """
        questions = self.file_manager.read()

        solutions = []
        for question in questions:
            self.logger.info(f"[START] question: {question.id}")

            answer = self.model.answer(question)

            question_entities = self.extractor.extract_entities(question.text)
            answer_entities = self.extractor.extract_entities(answer.text)

            question_entities = self.linker.link_entities_to_wikipedia(
                question_entities, answer.text
            )
            answer_entities = self.linker.link_entities_to_wikipedia(
                answer_entities, answer.text
            )

            answer_entities = self.extractor.clean_superfluous_entities(
                question_entities, answer_entities
            )

            problem = Problem(
                question=question,
                question_entities=question_entities,
                answer=answer,
                answer_entities=answer_entities,
            )

            extracted_answer = self.solver.solve(problem)
            correct = self.verifier.verify(problem, extracted_answer)
            # correct = self.verifier.verify(problem)
            solutions.append(
                Solution(
                    problem=problem,
                    extracted_answer=extracted_answer,
                    correct=correct,
                )
            )

            self.logger.info(f"[END] question: {question.id}")

        self.file_manager.write(solutions)


def resolve_entities():
    """
    Solves the assignment by reading an input file, resolving the entities, and writing the output.
    """
    Pipeline(
        file_manager=FileManager(
            input_path="data/input.txt", output_path="data/output.txt"
        ),
        model=LLM(model_path=f"models/llama-2-7b.Q4_K_M.gguf", stop=[], echo=False),
        extractor=EntityExtractor(),
        linker=EntityLinker(),
        solver=Solver(),
        verifier=Verifier(),
    ).resolve()


if __name__ == "__main__":
    init(autoreset=True)
    resolve_entities()
