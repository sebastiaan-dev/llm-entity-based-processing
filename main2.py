import spacy
from llama_cpp import Llama
from dataclasses import dataclass
import os


@dataclass
class Question:
    id: str
    text: str


@dataclass
class Answer:
    id: str
    text: str


@dataclass
class Entity:
    text: str
    label: str
    link: str = None


@dataclass
class Problem:
    question: Question
    question_entities: list[Entity]

    answer: Answer
    answer_entities: list[Entity]


@dataclass
class Solution:
    problem: Problem
    extracted_answer: str
    correct: bool


class DataManager:
    def __init__(self, input_path: str, output_path: str):
        """
        Initialize the data manager with the folder to the input and output data files.
        """
        self.input_path = input_path
        self.output_path = output_path

    def read(self) -> list[Question]:
        """
        Read file data of the format:

            <question-id><TAB><question-text><newline>

        Into a list of sorted Question objects.
        """
        questions = []

        with open(self.input_path, "r") as file:
            lines = file.readlines()

        for line in lines:
            if not line:
                continue
            try:
                id, text = line.split(maxsplit=1)
                questions.append(Question(id=id, text=text.replace("\n", "")))
            except ValueError:
                raise ValueError(f"Invalid line format: {line}")

        return questions

    def format_solution(self, solution: Solution) -> list[str]:
        """
        Format the solution into a list of strings.
        """
        lines = []

        problem = solution.problem
        q_id = problem.question.id

        lines.append(f'{q_id}\tR"{problem.question.text}"')
        lines.append(f'{q_id}\tA"{solution.extracted_answer}"')
        lines.append(f'{q_id}\tC"{"correct" if solution.correct else "incorrect"}"')

        for entity in list(problem.question_entities + problem.answer_entities):
            lines.append(f'{q_id}\tE"{entity.text}"\t"{entity.link}"')

        return lines

    def write(self, solutions: list[Solution]):
        """
        Write the solutions to the output file.
        """
        lines = []

        for solution in solutions:
            lines.extend(self.format_solution(solution))

        with open(self.output_path, "w") as file:
            for line in lines:
                file.write(line + "\n")


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

    def answer(self, question: Question) -> Answer:
        """
        Generate an answer to the given question with the LLM model.

        The result is stripped of leading and trailing whitespaces.
        """

        output = self.llm(
            question.text, max_tokens=self.max_tokens, stop=self.stop, echo=self.echo
        )

        return Answer(
            question.id, output["choices"][0]["text"].strip().replace("\n", " ")
        )


class EntityExtractor:
    def __init__(self):
        """
        Load a spaCy model for entity extraction.
        """
        self.nlp = spacy.load("en_core_web_trf")

    def extract_entities(self, text: str) -> list[Entity]:
        """
        Extract entities from the given text.
        """
        doc = self.nlp(text)
        entities = [Entity(ent.text, ent.label_) for ent in doc.ents]

        return entities


class EntityLinker:
    def __init__(self):
        pass

    def link_entity_to_wikipedia(self, entity: Entity) -> Entity:
        """
        Link an entity to its corresponding Wikipedia page.
        """
        entity.link = f"https://en.wikipedia.org/wiki/{entity.text.replace(' ', '_')}"

        return entity

    def link_entities_to_wikipedia(self, entities: list[Entity]) -> list[Entity]:
        """
        Link entities to their corresponding Wikipedia pages.
        """
        for entity in entities:
            entity.link = self.link_entity_to_wikipedia(entity).link


class Solver:
    def __init__(self):
        pass

    def solve(self, problem: Problem) -> str:
        """
        Solve the problem and return an answer.
        """
        return ""


class Verifier:
    def __init__(self):
        pass

    def verify(self, problem: Problem, extracted_answer: str) -> bool:
        """
        Verify the correctness of the extracted answer.
        """
        return True


class Pipeline:
    def __init__(
        self,
        data_manager: DataManager,
        model: LLM,
        extractor: EntityExtractor,
        linker: EntityLinker,
        solver: Solver,
        verifier: Verifier,
    ):
        self.data_manager = data_manager
        self.model = model
        self.extractor = extractor
        self.linker = linker
        self.solver = solver
        self.verifier = verifier

    def resolve(self):
        """
        This function acts as a pipeline for the different stages of the resolution process.

        1. Read the input data into a list of Question objects.
        2. Generate the answers for each question using an LLM.
        3. Extract entities from the questions and answers.
        4. Extract the formatted answer from the generated answer.
        5. Check the correctness of the extracted answer.
        """
        questions = self.data_manager.read()

        solutions = []
        for question in questions:
            answer = self.model.answer(question)

            question_entities = self.extractor.extract_entities(question.text)
            answer_entities = self.extractor.extract_entities(answer.text)

            self.linker.link_entities_to_wikipedia(question_entities)
            self.linker.link_entities_to_wikipedia(answer_entities)

            problem = Problem(
                question=question,
                question_entities=question_entities,
                answer=answer,
                answer_entities=answer_entities,
            )

            extracted_answer = self.solver.solve(problem)
            correct = self.verifier.verify(problem, extracted_answer)

            solutions.append(
                Solution(
                    problem=problem,
                    extracted_answer=extracted_answer,
                    correct=correct,
                )
            )

        self.data_manager.write(solutions)


def resolve_entities():
    """
    Solves the assignment by reading an input file, resolving the entities, and writing the output.
    """
    Pipeline(
        data_manager=DataManager(
            input_path="data/input.txt", output_path="data/output.txt"
        ),
        model=LLM(model_path=f"models/llama-2-7b.Q4_K_M.gguf", stop=[], echo=False),
        extractor=EntityExtractor(),
        linker=EntityLinker(),
        solver=Solver(),
        verifier=Verifier(),
    ).resolve()


if __name__ == "__main__":
    resolve_entities()
