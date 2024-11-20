import spacy

from abc import ABC, abstractmethod


class Extractor(ABC):
    @abstractmethod
    def extract_entities(self, text: str) -> str:
        pass


class SpacyExtractor(Extractor):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")

    def extract_entities(self, text: str) -> str:
        doc = self.nlp(text)
        return doc
