import logging
import spacy
from debug import print_debug, print_info
from models import Entity


class EntityExtractor:
    def __init__(self):
        """
        Load a spaCy model for entity extraction.
        """
        self.nlp = spacy.load("en_core_web_trf")

        self.logger = logging.getLogger(__name__)

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
        self.logger.info(f"entities: {[(ent.text, ent.label_) for ent in doc.ents]}")

        entities = self.entity_deduplication(
            [Entity(ent.text, ent.label_) for ent in doc.ents]
        )
        self.logger.info(
            f"deduped entities: {[(ent.text, ent.label) for ent in entities]}"
        )

        print_info(
            "Extracted entities:", ", ".join([entity.text for entity in entities])
        )

        return entities

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
