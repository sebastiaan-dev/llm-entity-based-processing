import requests as req
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util

from models import Entity, Problem
from debug import print_debug, print_warn
import re


class Verifier:
    def __init__(self):
        self.similarity_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    def normalize_question(self, question: str) -> str:

        q = question.strip()
        if q.lower().startswith("question:"):
            q = q[len("question:") :].strip()
        return q

    def query_wikipedia(self, question: str, entities: list[Entity]) -> str:
        """
        Query Wikipedia for relevant results based on question context and extracted entities.
        """
        # Construct query with entities and keywords
        question_keywords = self.normalize_question(question).split()
        context_keywords = " ".join(
            question_keywords[:5]
        )  # Focus on first few keywords
        entity_terms = " ".join([entity.text for entity in entities])
        main_query = f"{entity_terms} {context_keywords}"

        # print(f"DEBUG: Querying Wikipedia with: {main_query}")

        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": main_query,
            "format": "json",
        }

        response = req.get(url, params=params)
        data = response.json()

        # Extract the most relevant title
        results = data.get("query", {}).get("search", [])
        best_match = None
        best_score = 0
        for result in results:
            title = result["title"].lower()
            score = fuzz.partial_ratio(main_query.lower(), title)
            # print(f"DEBUG: Comparing '{main_query}' with '{title}', score: {score}")
            if score > best_score:
                best_match = title
                best_score = score

        # print(f"DEBUG: Best match: {best_match} with score: {best_score}")
        return best_match if best_score > 70 else None

    def verify(self, problem: Problem, extracted_answer: str) -> bool:
        """
        Verify the correctness of the extracted answer.

        PROBLEMS:

        expected_title and extracted_anwer are incorrect we have to use other information we have

        We could check if answer is yes/no then retrive the link from the entity from the Q and
        calculate the score based on this.

        If answer is the link, check the score based on the link and the entity from the Q
        """

        # if len(problem.question_entities) == 0:
        #    return False

        question = self.normalize_question(problem.question.text)

        # Query Wikipedia for the expected title
        expected_title = self.query_wikipedia(
            problem.question.text, problem.question_entities
        )

        # Normalize the extracted answer entity
        extracted_entity = extracted_answer.split("/")[-1].replace("_", " ").lower()

        print_debug("Extracted entity: ", extracted_entity)
        print_debug("Expected title: ", expected_title)

        # Fallback to context validation if extracted entity is the link
        link = problem.answer.text
        if link and link.startswith("https"):
            try:
                response = req.get(link)
                if response.status_code == 200:
                    # Validate context relevance
                    question_terms = [term.lower() for term in question.split()]
                    if any(term in response.text.lower() for term in question_terms):
                        # print("DEBUG: Correctness Comparison: correct (fallback context validation)")
                        return True
            except Exception as e:
                print(f"DEBUG: Error accessing Wikipedia page for fallback: {e}")
            return False
        else:
            # its a yes no answer still to implement
            return True

        if not problem.question_entities:
            return False

        if not extracted_entity:
            # print("DEBUG: Extracted Entity is None. Skipping similarity checks.")
            return False

        if not expected_title and extracted_entity:
            print_debug("Expected entity is: ", extracted_entity)
            entity_context_title = self.query_wikipedia(extracted_entity, [])
            if entity_context_title:
                fallback_url = f"https://en.wikipedia.org/wiki/{entity_context_title.replace(' ', '_')}"
                try:
                    response = req.get(fallback_url)
                    if response.status_code == 200:
                        # Validate context relevance
                        question_terms = [term.lower() for term in question.split()]
                        if any(
                            term in response.text.lower() for term in question_terms
                        ):
                            # print("DEBUG: Correctness Comparison: correct (fallback context validation)")
                            return True
                except Exception as e:
                    print(f"DEBUG: Error accessing Wikipedia page for fallback: {e}")
            return False

        # Direct match: Extracted entity matches expected title
        # if fuzz.partial_ratio(extracted_entity, expected_title.lower()) > 70:
        # print("DEBUG: Correctness Comparison: correct (direct match)")
        #    return True

        # Semantic similarity match
        similarity_score = util.pytorch_cos_sim(
            self.similarity_model.encode(extracted_entity, convert_to_tensor=True),
            self.similarity_model.encode(expected_title, convert_to_tensor=True),
        ).item()

        if similarity_score > 0.7:
            # print("DEBUG: Correctness Comparison: correct (semantic similarity)")
            return True

        # Validate extracted entity within the context of the expected title
        wikipedia_url = (
            f"https://en.wikipedia.org/wiki/{expected_title.replace(' ', '_')}"
        )
        try:
            response = req.get(wikipedia_url)
            if (
                response.status_code == 200
                and extracted_entity in response.text.lower()
            ):
                # print("DEBUG: Correctness Comparison: correct (entity found in page)")
                return True
        except Exception as e:
            print(f"DEBUG: Error accessing Wikipedia page: {e}")

        # Fallback: Query Wikipedia for the extracted entity and validate it
        entity_context_title = self.query_wikipedia(extracted_entity, [])
        if entity_context_title:
            fallback_url = f"https://en.wikipedia.org/wiki/{entity_context_title.replace(' ', '_')}"
            try:
                response = req.get(fallback_url)
                if response.status_code == 200:
                    # Check if question context appears in the Wikipedia page
                    question_terms = [term.lower() for term in question.split()]
                    if any(term in response.text.lower() for term in question_terms):
                        # print("DEBUG: Correctness Comparison: correct (fallback context validation)")
                        return True
            except Exception as e:
                print(f"DEBUG: Error accessing Wikipedia page for fallback: {e}")

        # print("DEBUG: Correctness Comparison: incorrect")
        return False
