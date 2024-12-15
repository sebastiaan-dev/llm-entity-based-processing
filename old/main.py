import spacy
import requests
from fuzzywuzzy import fuzz
from abc import ABC, abstractmethod
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer, util


class Model(ABC):
    @abstractmethod
    def answer(self, question: str) -> str:
        pass


class LlamaModel(Model):
    def __init__(
        self,
        model_path: str,
        max_tokens: int = 10,
        stop: list = ["Q:", "\n"],
        echo: bool = True,
    ):
        self.llm = Llama(model_path=f"models/{model_path}", verbose=False)
        self.max_tokens = max_tokens
        self.stop = stop
        self.echo = echo
        self.nlp = spacy.load("en_core_web_trf")
        self.valid_labels = [
            "GPE",
            "ORG",
            "LOC",
            "PERSON",
            "EVENT",
            "WORK_OF_ART",
            "NORP",
            "FAC",
            "LAW",
            "DATE",
            "NUMBER",
            "QUANTITY",
            "PRODUCT",
            "LANGUAGE",
        ]
        self.similarity_model = SentenceTransformer("all-MiniLM-L6-v2")

    def answer(self, prompt: str) -> str:
        response = self.llm(
            prompt, max_tokens=self.max_tokens, stop=self.stop, echo=self.echo
        )
        raw_text = response["choices"][0]["text"].strip()
        return raw_text

    def extract_entities(self, text: str):
        # This function will be used on the Q and A
        doc = self.nlp(text)
        doc_text = doc.text
        print(f"DOC:{doc}")
        # take the first part of the answer until the . to make the retrival of the entities more precise

        if "." in doc_text:
            doc_text = doc_text.split(".", 1)[0].strip()
            print(f"DOC_STRIPPED:{doc}")
        doc = self.nlp(doc_text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in self.valid_labels:
                e_clean = ent.text.strip().rstrip(".")
                # Remove leading "the" (case insensitive)
                # This was needed because wikidata was giving us the wrong links if "The" was included in the request of the entity
                if e_clean.lower().startswith("the "):
                    e_clean = e_clean[4:]  # Remove "the" and the space

                entities.append((e_clean, ent.label_))
        print(f"ENTITIES: {entities}")
        return entities

    def normalize_question(self, question: str) -> str:
        q = question.strip()
        if q.lower().startswith("question:"):
            q = q[len("question:") :].strip()
        return q

    def is_yes_no_question(self, question: str) -> bool:
        q_norm = self.normalize_question(question).lower()
        return q_norm.startswith(("is ", "are ", "does ", "do ", "can ", "did "))

    def extract_answer(self, A: str, B: str) -> str:
        if self.is_yes_no_question(A):
            b_lower = B.lower()
            if "yes" in b_lower:
                return "yes"
            elif "no" in b_lower:
                return "no"
            return "unknown"

        if "Answer:" in B:
            answer_part = B.split("Answer:", 1)[1].strip()
            entities = self.extract_entities(answer_part)
            print(f"ENT: {entities} A_PART: {answer_part}")
            if entities:
                res = get_url(entities[0][0])
                print(f"{entities[0][0]} -> {res}")
                return res
            else:
                return get_url(answer_part)
                # return answer_part.split("\n")[0].strip()

        entities = self.extract_entities(B)
        if entities:
            res = get_url(entities[0][0])
            print(f"{entities[0][0]} -> {res}")
            # return f"https://en.wikipedia.org/wiki/{entities[0][0].replace(' ', '_')}"
            return res
        return "unknown"

    def query_wikipedia(self, question: str, entities: list) -> str:
        """
        Query Wikipedia for relevant results based on question context and extracted entities.
        """
        # Construct query with entities and keywords
        question_keywords = self.normalize_question(question).split()
        context_keywords = " ".join(
            question_keywords[:5]
        )  # Focus on first few keywords
        entity_terms = " ".join([entity[0] for entity in entities])
        main_query = f"{entity_terms} {context_keywords}"

        # print(f"DEBUG: Querying Wikipedia with: {main_query}")

        url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": main_query,
            "format": "json",
        }
        response = requests.get(url, params=params)
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

    def check_correctness(self, A: str, extracted_answer: str) -> str:
        """
        Validate extracted answers using Wikipedia results, question context,
        and entity relationships.
        """
        question = self.normalize_question(A)
        # in here we are using again the extract entities on the Q, when we have already the extracted data, just reuse that one
        entities = self.extract_entities(A)
        if not entities:
            # print("DEBUG: No entities found in the question.")
            return "unknown"

        # Query Wikipedia for the expected title
        expected_title = self.query_wikipedia(A, entities)

        # Normalize the extracted answer entity
        extracted_entity = extracted_answer.split("/")[-1].replace("_", " ").lower()

        # Debugging details
        # print(f"DEBUG: Question: {question}")
        # print(f"DEBUG: Expected Title: {expected_title}")
        # print(f"DEBUG: Extracted Entity: {extracted_entity}")

        # Handle cases where expected_title is None
        if not expected_title:
            # print("DEBUG: Expected Title is None. Skipping similarity checks.")
            # Fallback to context validation if extracted entity is valid
            if extracted_entity:
                entity_context_title = self.query_wikipedia(extracted_entity, [])
                if entity_context_title:
                    fallback_url = f"https://en.wikipedia.org/wiki/{entity_context_title.replace(' ', '_')}"
                    try:
                        response = requests.get(fallback_url)
                        if response.status_code == 200:
                            # Validate context relevance
                            question_terms = [term.lower() for term in question.split()]
                            if any(
                                term in response.text.lower() for term in question_terms
                            ):
                                # print("DEBUG: Correctness Comparison: correct (fallback context validation)")
                                return "correct"
                    except Exception as e:
                        print(
                            f"DEBUG: Error accessing Wikipedia page for fallback: {e}"
                        )
            return "incorrect"

        # Direct match: Extracted entity matches expected title
        if fuzz.partial_ratio(extracted_entity, expected_title.lower()) > 70:
            # print("DEBUG: Correctness Comparison: correct (direct match)")
            return "correct"

        # Semantic similarity match
        similarity_score = util.pytorch_cos_sim(
            self.similarity_model.encode(extracted_entity, convert_to_tensor=True),
            self.similarity_model.encode(expected_title, convert_to_tensor=True),
        ).item()

        if similarity_score > 0.7:
            # print("DEBUG: Correctness Comparison: correct (semantic similarity)")
            return "correct"

        # Validate extracted entity within the context of the expected title
        wikipedia_url = (
            f"https://en.wikipedia.org/wiki/{expected_title.replace(' ', '_')}"
        )
        try:
            response = requests.get(wikipedia_url)
            if (
                response.status_code == 200
                and extracted_entity in response.text.lower()
            ):
                # print("DEBUG: Correctness Comparison: correct (entity found in page)")
                return "correct"
        except Exception as e:
            print(f"DEBUG: Error accessing Wikipedia page: {e}")

        # Fallback: Query Wikipedia for the extracted entity and validate it
        entity_context_title = self.query_wikipedia(extracted_entity, [])
        if entity_context_title:
            fallback_url = f"https://en.wikipedia.org/wiki/{entity_context_title.replace(' ', '_')}"
            try:
                response = requests.get(fallback_url)
                if response.status_code == 200:
                    # Check if question context appears in the Wikipedia page
                    question_terms = [term.lower() for term in question.split()]
                    if any(term in response.text.lower() for term in question_terms):
                        # print("DEBUG: Correctness Comparison: correct (fallback context validation)")
                        return "correct"
            except Exception as e:
                print(f"DEBUG: Error accessing Wikipedia page for fallback: {e}")

        # print("DEBUG: Correctness Comparison: incorrect")
        return "incorrect"

    def link_entities_to_wiki(self, entities):
        linked = []
        print(f"TEST:{entities}")
        for e in entities:
            url = get_url(e[0])  # Call get_url with the current entity
            # print quest_id, the entity and the corresponding url
            print(f'E"{e[0]}"\t"{url}"')
            linked.append((e[0], url))
        return linked


def get_url(entity):
    url = "https://wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": entity,
    }
    # make the request to the wikidata api with the corresponding parameters
    response = requests.get(url, params=params)
    if response.status_code == 200:
        # save the data from the response
        data = response.json()
        # print(data)
        if data["search"]:
            # get the Wikidata entity ID from the search result
            entity_id = data["search"][0]["id"]

            # entity ID used to get detailed information from Wikidata
            entity_url = f"https://www.wikidata.org/wiki/{entity_id}"
            details_url = f"https://www.wikidata.org/w/api.php"

            # get the Wikipedia links
            params = {
                "action": "wbgetentities",
                "ids": entity_id,
                "sites": "enwiki",  # the English Wikipedia link
                "props": "sitelinks",  # we need sitelinks to retrive the original link
                "format": "json",
            }

            response = requests.get(details_url, params=params)
            data = response.json()

            # check if the Wikipedia link exists in the response from the json
            if "entities" in data and entity_id in data["entities"]:
                entity_data = data["entities"][entity_id]

                if "sitelinks" in entity_data and "enwiki" in entity_data["sitelinks"]:
                    wikipedia_title = entity_data["sitelinks"]["enwiki"]["title"]
                    wikipedia_link = f"https://en.wikipedia.org/wiki/{wikipedia_title}"
                    return wikipedia_link

    return None


def main():
    model = LlamaModel(model_path="/llama-2-7b.Q4_K_M.gguf")

    queries = []
    with open("input.txt", "r") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                q_id, A = parts
            else:
                q_id = "Q"
                A = line
            queries.append((q_id, A))

    output_lines = []
    for q_id, A in queries:
        B = model.answer(A)

        ents_A = model.extract_entities(A)
        ents_B = model.extract_entities(B)
        all_ents = list(set(ents_A + ents_B))

        extracted_answer = model.extract_answer(A, B)
        correctness = model.check_correctness(A, extracted_answer)

        output_lines.append(f'{q_id}\tR"{B}"')
        output_lines.append(f'{q_id}\tA"{extracted_answer}"')
        output_lines.append(f'{q_id}\tC"{correctness}"')

        linked_ents = model.link_entities_to_wiki(all_ents)
        for ent, url in linked_ents:
            output_lines.append(f'{q_id}\tE"{ent}"\t"{url}"')
    # TODO: add case where output does not exist and it has to be created automatically
    with open("output.txt", "w") as outfile:
        for line in output_lines:
            outfile.write(line + "\n")


if __name__ == "__main__":
    main()
