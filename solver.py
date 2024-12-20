from typing import Any
import joblib
import requests as req

from debug import print_debug, print_error, print_info, print_warn
from models import Entity, Problem


# TODO: Use SVM to classify the question type and handle it accordingly.
# TODO: If cannot decide we should fall back to the more popular entity.
# TODO: Use aliases to improve entity matching.


class Solver:
    def __init__(self):
        self.wikipedia_api = "https://en.wikipedia.org/w/api.php"
        self.model = joblib.load("regression/best_model.joblib")
        self.label_encoder = joblib.load("regression/label_encoder.joblib")

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
        q = question.strip().lower()

        if q.startswith("question:"):
            q = q[len("question:") :].strip()

        if "answer:" in q:
            before_answer = q[: q.index("answer:")].strip()
            after_answer = q[q.index("answer:") + len("answer:") :].strip()
            q = f"{before_answer} {after_answer}"

        return q

    def is_yes_no_question(self, question: str) -> bool:
        """
        Verify the question is of the expected format for a yes/no question.
        """
        q_norm = self.normalize_question(question)
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

    def get_wikipedia_wordcount(self, entity: Entity, word: str) -> tuple[int, int]:
        """
        Get the word count of a Wikipedia page.
        """
        params = {
            "action": "query",
            "titles": entity.text,
            "prop": "extracts",
            "format": "json",
            "explaintext": True,
            "redirects": "1",
        }

        data = self.query_wikipedia(params)
        page = list(data["query"]["pages"].values())[0]
        extract = page.get("extract", "")

        return extract.lower().count(word.lower()), len(extract.split())

    def get_counts_a_in_q(self, problem: Problem) -> dict[str, int]:
        a_in_q_distribution = {}

        for a_entity in problem.answer_entities:
            total_word_count = 0

            for q_entity in problem.question_entities:
                word_count, len = self.get_wikipedia_wordcount(a_entity, q_entity.text)
                normalized_word_count = word_count / len
                total_word_count += normalized_word_count

            a_in_q_distribution[a_entity.text] = total_word_count

        print_debug(
            "A -> Q Distribution:",
            [f"{k}: {v}" for k, v in a_in_q_distribution.items()],
        )

        return a_in_q_distribution

    def get_counts_q_in_a(self, problem: Problem) -> dict[str, int]:
        q_in_a_distribution = {}

        for q_entity in problem.question_entities:
            for a_entity in problem.answer_entities:
                word_count, len = self.get_wikipedia_wordcount(q_entity, a_entity.text)
                normalized_word_count = word_count / len

                if (q_entity.text, a_entity.text) not in q_in_a_distribution:
                    q_in_a_distribution[a_entity.text] = normalized_word_count
                else:
                    q_in_a_distribution[a_entity.text] += normalized_word_count

        print_debug(
            "Q -> A Distribution:",
            [f"{k}: {v}" for k, v in q_in_a_distribution.items()],
        )

        return q_in_a_distribution

    def get_ranked_by_wordcount(self, problem: Problem) -> list[Entity]:
        a_in_q_distribution = self.get_counts_a_in_q(problem)
        q_in_a_distribution = self.get_counts_q_in_a(problem)

        total_distribution = {}

        # Combine the distributions.
        for entity, count in a_in_q_distribution.items():
            total_distribution[entity] = count

        for entity, count in q_in_a_distribution.items():
            if entity in total_distribution:
                total_distribution[entity] += count
            else:
                total_distribution[entity] = count

        ranked = sorted(
            problem.answer_entities,
            key=lambda entity: total_distribution[entity.text],
            reverse=True,
        )

        nonzero = [entity for entity in ranked if total_distribution[entity.text] > 0]

        return nonzero

    def preprocess_question(self, q: str) -> str:
        return q.lower().strip()

    def classify_question(self, question):
        preprocessed = self.preprocess_question(question)
        predicted_labels = self.model.predict_proba([preprocessed])[0]

        # Get the top 3 labels
        top_indices = predicted_labels.argsort()[::-1][:3]
        top_labels = self.label_encoder.inverse_transform(top_indices)

        top_confidences = predicted_labels[top_indices]
        top_predictions = list(zip(top_labels, top_confidences))
        print_debug(f"Predicted probabilities:", top_predictions)

        return top_labels

    def derive_question_type(self, question: str) -> list[str]:
        """
        Derive the question entity target type from the question text.

        Returns the entity label (should be the same as the labels used in the recognition pipeline).
        """
        q = self.normalize_question(question)

        if q.startswith("who"):
            return ["PERSON"]
        elif q.startswith("where"):
            return ["LOC"]
        elif q.startswith("when"):
            return ["DATE"]

        # We could not derive the question type using regex, so we use a classifier.
        return self.classify_question(q)

    def try_label_order_n(
        self, ranked_entities: list[Entity], predicted_label: str
    ) -> str | None:
        for entity in ranked_entities:
            # print_debug(f"Checking entity:", entity.text)
            # print_debug(f"Predicted label:", entity.label)
            # print_debug(f"Predicted label:", predicted_label)
            if entity.label == predicted_label:
                print_info(f"Selected entity:", entity.text)
                return entity.link

        return None

    def handle_open_question(self, problem: Problem) -> str | None:
        """
        Handles open questions.
        """
        # No entities found means we cannot solve the problem.
        if len(problem.answer_entities) == 0:
            print_error("No entities found, cannot solve the problem.")
            return ""

        # Handle the edge case where we found only 1 entity, there is nothing more to filter
        # and we can return the link directly.
        if len(problem.answer_entities) == 1:
            print_info("Only 1 entity found, returning the link directly.")
            return problem.answer_entities[0].link

        print_debug(
            "Answer entities:",
            [(entity.text, entity.link) for entity in problem.answer_entities],
        )
        print_debug(
            "Question entities:",
            [(entity.text, entity.link) for entity in problem.question_entities],
        )

        # Get the ranked entities by word count.
        ranked_entities = self.get_ranked_by_wordcount(problem)
        predicted_labels = self.derive_question_type(problem.question.text)

        if len(ranked_entities) == 0:
            print_error("No entities found, cannot solve the problem.")
            return None

        # Find the entity with the highest rank that matches the predicted label.
        for predicted_label in predicted_labels:
            link = self.try_label_order_n(ranked_entities, predicted_label)

            if link:
                return link

        # If we could not find an entity with the predicted label, return the link of the highest ranked entity.
        print_warn(
            f"Could not find an entity with the predicted label:", predicted_labels
        )
        print_info(
            f"Entities with label:",
            [(entity.text, entity.label) for entity in ranked_entities],
        )
        print_info(
            f"Returning the link of the highest ranked entity.", ranked_entities[0].text
        )

        return ranked_entities[0].link

    def solve(self, problem: Problem) -> str:
        """
        Try to solve the answer by classifying the question type and handling it accordingly.
        """
        if self.is_yes_no_question(problem.question.text):
            return self.handle_yes_no_question(problem.answer.text)

        return self.handle_open_question(problem)
