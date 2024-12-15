import spacy
from debug import print_debug, print_info
from models import Entity


class EntityExtractor:
    def __init__(self):
        """
        Load a spaCy model for entity extraction.
        """
        self.nlp = spacy.load("en_core_web_trf")

    def is_similar(self, seen_list: set, tag: str) -> bool:
        """
        Check if two strings are similar.
        """
        for seen_tag in seen_list:
            if tag in seen_tag:
                return True

        return False

    def entity_deduplication(self, entities: list[Entity]) -> list[Entity]:
        """
        Check for duplicates inside entities list
        """

        seen = set()
        stripped = set()
        final_entities = []

        for entity in sorted(entities, key=lambda e: len(e.text), reverse=True):
            if not self.is_similar(seen, entity.text.lower()):
                seen.add(entity.text.lower())
                final_entities.append(entity)
            else:
                stripped.add(entity.text)

        if len(stripped) > 0:
            print_debug("Stripped entities:", ", ".join(stripped))

        return final_entities

    def entity_postprocessing(self, entities: list[Entity]) -> list[Entity]:
        """
        Postprocess entities to improve entity extraction.
        """
        cleaned_entities = []

        for entity in entities:
            text = entity.text.strip()
            # Remove leading "the" (case insensitive)
            # This was needed because wikidata was giving us the wrong links if "The" was included in the request of the entity
            if text.lower().startswith("the "):
                text = text[4:]  # Remove "the" and the space

            cleaned_entities.append(Entity(text, entity.label))

        return self.entity_postprocessing(entities)

    def extract_entities(self, text: str) -> list[Entity]:
        """
        Extract entities from the given text.
        """
        # TODO: Try to use capitalization to improve entity extraction

        doc = self.nlp(text)
        entities = self.entity_deduplication(
            [Entity(ent.text, ent.label_) for ent in doc.ents]
        )

        print_info(
            "Extracted entities:", ", ".join([entity.text for entity in entities])
        )

        return entities
