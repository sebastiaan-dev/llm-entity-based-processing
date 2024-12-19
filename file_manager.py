from models import Question, Solution


class FileManager:
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
