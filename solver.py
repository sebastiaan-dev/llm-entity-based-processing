from debug import print_debug
from models import Entity, Problem


class Solver:
    def __init__(self):
        pass

    def normalize_question(self, question: str) -> str:
        """
        Remove possible prefixes from the question for correct downstream parsing.
        """
        q = question.strip()
        if q.lower().startswith("question:"):
            q = q[len("question:") :].strip()
        return q

    def is_yes_no_question(self, question: str) -> bool:
        """
        Verify the question is of the expected format for a yes/no question.
        """
        q_norm = self.normalize_question(question).lower()
        return q_norm.startswith(("is ", "are ", "does ", "do ", "can ", "did "))

    def handle_yes_no_question(self, answer: str) -> str:
        """
        Handles yes/no questions.
        """
        if "yes" in answer:
            return "Yes"
        elif "no" in answer:
            return "No"
        else:
            raise ValueError(f"Invalid answer for yes/no question: {answer}")

    def clean_superfluous_entities(
        self, question_entities: list[Entity], answer_entities: list[Entity]
    ) -> list[Entity]:
        """
        Remove entities that are present in the question from the answer entities,
        as this entity is already known from the question it would not make sense to use it for the answer.
        """
        removed_entities = []
        cleaned_entities = []

        for a_entity in answer_entities:
            duplicate = False

            for q_entity in question_entities:
                if a_entity.link.lower() == q_entity.link.lower():
                    duplicate = True
                    removed_entities.append(a_entity)
                    break

            if not duplicate:
                cleaned_entities.append(a_entity)

        if len(removed_entities) > 0:
            print_debug(
                "Removed superfluous entities:",
                ", ".join([e.text for e in removed_entities]),
            )

        return cleaned_entities

    def handle_open_question(self, problem: Problem) -> str:
        """
        Handles open questions.
        """
        problem.answer_entities = self.clean_superfluous_entities(
            problem.question_entities, problem.answer_entities
        )

        # Handle the edge case where we found only 1 entity, there is nothing more to filter
        # and we can return the link directly.
        if len(problem.answer_entities) == 1:
            return problem.answer_entities[0].link

        return ""

    def solve(self, problem: Problem) -> str:
        """
        Try to solve the answer by classifying the question type and handling it accordingly.
        """
        if self.is_yes_no_question(problem.question.text):
            return self.handle_yes_no_question(problem.answer.text)

        return self.handle_open_question(problem)
