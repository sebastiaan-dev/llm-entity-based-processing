from typing import Any
import requests as req

from debug import print_debug, print_info
from models import Entity, Problem


# TODO: Use SVM to classify the question type and handle it accordingly.
# TODO: If cannot decide we should fall back to the more popular entity.


class Solver:
    def __init__(self):
        self.wikipedia_api = "https://en.wikipedia.org/w/api.php"

    def query_wikipedia(self, query: Any) -> Any:
        response = req.get(self.wikipedia_api, params=query)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to query Wikipedia: {response.status_code}")

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
                # If they have equal semantic meaning, remove the answer entity.
                if a_entity.link.lower() == q_entity.link.lower():
                    duplicate = True
                    removed_entities.append(a_entity)
                    break
                # A more aggressive approach is expressed here, if the answer entity is a substring of the question entity,
                # it is removed as well.
                if a_entity.text.lower() in q_entity.text.lower():
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

    def get_wikipedia_wordcount(self, entity: Entity, word: str) -> int:
        """
        Get the word count of a Wikipedia page.
        """
        params = {
            "action": "query",
            "titles": entity.text,
            "prop": "extracts",
            "format": "json",
            "explaintext": True,
        }

        data = self.query_wikipedia(params)
        page = list(data["query"]["pages"].values())[0]
        extract = page.get("extract", "")

        return extract.lower().count(word.lower())

    def handle_open_question(self, problem: Problem) -> str:
        """
        Handles open questions.
        """
        problem.answer_entities = self.clean_superfluous_entities(
            problem.question_entities, problem.answer_entities
        )

        # No entities found means we cannot solve the problem.
        if len(problem.answer_entities) == 0:
            return ""

        # Handle the edge case where we found only 1 entity, there is nothing more to filter
        # and we can return the link directly.
        if len(problem.answer_entities) == 1:
            return problem.answer_entities[0].link

        distribution = {}

        for a_entity in problem.answer_entities:
            total_word_count = 0

            for q_entity in problem.question_entities:
                word_count = self.get_wikipedia_wordcount(a_entity, q_entity.text)
                total_word_count += word_count

            distribution[a_entity.text] = total_word_count

        print_debug("Distribution:", [f"{k}: {v}" for k, v in distribution.items()])

        max_text = max(distribution, key=distribution.get)
        # Return the link of the entity with the highest word count.
        for entity in problem.answer_entities:
            if entity.text == max_text:
                print_info(f"Selected entity:", entity.text)
                return entity.link

    def solve(self, problem: Problem) -> str:
        """
        Try to solve the answer by classifying the question type and handling it accordingly.
        """
        if self.is_yes_no_question(problem.question.text):
            return self.handle_yes_no_question(problem.answer.text)

        return self.handle_open_question(problem)
