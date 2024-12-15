from dataclasses import dataclass


@dataclass
class Question:
    id: str
    text: str


@dataclass
class Answer:
    id: str
    text: str


@dataclass
class Entity:
    text: str
    label: str
    link: str = None


@dataclass
class Problem:
    question: Question
    question_entities: list[Entity]

    answer: Answer
    answer_entities: list[Entity]


@dataclass
class Solution:
    problem: Problem
    extracted_answer: str
    correct: bool


@dataclass
class WikidataResult:
    id: str
    label: str
    description: str
    link: str
