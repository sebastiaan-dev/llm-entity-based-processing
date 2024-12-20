import numpy as np
import requests as req
from typing import Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from debug import print_debug, print_warn
from models import Entity, WikidataResult


# TODO: We might want to compare pages instead of descriptions. This gives more context.


class EntityLinker:
    def __init__(self):
        self.wikipedia_api = "https://en.wikipedia.org/w/api.php"
        self.wikidata_api = "https://wikidata.org/w/api.php"

    def query_wikidata(self, query: Any) -> Any:
        response = req.get(self.wikidata_api, params=query)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to query Wikidata: {response.status_code}")

    def query_wikipedia(self, query: Any) -> Any:
        response = req.get(self.wikipedia_api, params=query)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to query Wikipedia: {response.status_code}")

    def try_direct_link(self, entity: Entity) -> str | None:
        """
        Try to link an entity directly to its Wikipedia page.
        """
        params = {
            "action": "query",
            "titles": entity.text,
            "format": "json",
            "redirects": "1",
            "prop": "pageprops",
        }

        data = self.query_wikipedia(params)
        page = list(data["query"]["pages"].values())[0]

        if "missing" in page:
            print_debug("Malformed link, falling back:", entity.text)
            return None

        if "disambiguation" in page["pageprops"]:
            print_debug("Found disambiguation link, falling back:", entity.text)
            return None

        return f"https://en.wikipedia.org/wiki/{entity.text.replace(' ', '_')}"

    def get_wikidata_matches(self, entity: Entity) -> list[WikidataResult]:
        """
        We make use of WikiData's relevancy ranking when querying for possible entity matches.
        """
        params = {
            "action": "wbsearchentities",
            "language": "en",
            "format": "json",
            "search": entity.text,
            "limit": 10,
            "props": "url",
        }

        data = self.query_wikidata(params)

        wikidata_results = []
        for result in data["search"]:
            # Currently, we skip entities which do not have a description.
            # These are usually less relevant results, and are redirects or instances of entities with descriptions.
            if "description" not in result:
                continue

            wikidata_results.append(
                WikidataResult(
                    id=result["id"],
                    label=result["label"],
                    description=result["description"],
                    link=result["url"],
                )
            )

        return wikidata_results

    def get_wikidata_sitelink(self, entity_id: str) -> str | None:
        """
        Get the Wikipedia link from the corresponding Wikidata entity.
        """

        params = {
            "action": "wbgetentities",
            "ids": entity_id,
            "sites": "enwiki",
            "props": "sitelinks",
            "format": "json",
        }

        data = self.query_wikidata(params)

        if not "sitelinks" in data["entities"][entity_id]:
            return None

        if not "enwiki" in data["entities"][entity_id]["sitelinks"]:
            return None

        title = data["entities"][entity_id]["sitelinks"]["enwiki"]["title"]

        return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

    def link_entity_to_wikipedia(
        self, entity: Entity, answer_text: str
    ) -> Entity | None:
        """
        Link an entity to its corresponding Wikipedia page.
        """
        direct_link = self.try_direct_link(entity)
        if direct_link:
            print_debug("Found direct link for entity:", entity.text)
            return direct_link

        wikidata_results = self.get_wikidata_matches(entity)

        if not wikidata_results:
            print_warn("Failed to find Wikidata results for entity:", entity.text)
            return None

        # https://medium.com/@gusainanurag58/tf-idf-vectorizer-explained-373b3f07d23b
        # https://openclassrooms.com/en/courses/6532301-introduction-to-natural-language-processing/8081363-apply-the-tf-idf-vectorization-approach
        vectorizer = TfidfVectorizer()
        corpus = [answer_text] + [result.description for result in wikidata_results]
        tfdif = vectorizer.fit_transform(corpus)

        # Compare the description of the answer with the descriptions of the wikidata results.
        similarities = cosine_similarity(tfdif[0:1], tfdif[1:]).flatten()
        ranked_idx = np.flip(np.argsort(similarities))
        ranked_results = [wikidata_results[i] for i in ranked_idx]

        link = None
        for result in ranked_results:
            link = self.get_wikidata_sitelink(result.id)

            if link:
                print_debug(
                    "Found WikiData-based link for entity:",
                    f"{result.label}, {result.description}",
                )
                break

            print_warn(
                "Failed to find a link for entity:",
                f"{result.label}, {result.description}",
            )

        return link

    def clean_unlinkable_entities(self, entities: list[Entity]) -> list[Entity]:
        """
        Remove entities that cannot be linked to Wikipedia.
        """
        cleaned_entities = []

        for entity in entities:
            if entity.link:
                cleaned_entities.append(entity)
            else:
                print_debug(
                    "Stripping unlinkable entity:", f"{entity.label}, {entity.text}"
                )

        return cleaned_entities

    def link_entities_to_wikipedia(
        self, entities: list[Entity], answer_text: str
    ) -> list[Entity]:
        """
        Link entities to their corresponding Wikipedia pages.
        """
        for entity in entities:
            entity.link = self.link_entity_to_wikipedia(entity, answer_text)

        return self.clean_unlinkable_entities(entities)
