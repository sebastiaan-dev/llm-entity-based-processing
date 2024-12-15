import os

from extractor import SpacyExtractor
from llm import LlamaModel
from parser import TxtParser
from pprint import pprint


def generate_answers(questions, parser):
    model = LlamaModel(model_path="llama-2-7b.Q4_K_M.gguf", stop=["\n"], echo=False)
    # Check if the file exists at data/answers.txt.
    if not os.path.exists("data/answers.txt"):
        # If it does not exist we generate the answers.
        with open("data/answers.txt", "w") as f:
            for idx, question in enumerate(questions, start=1):
                answer = model.answer(question["text"])
                f.write(f"Q{idx}\t{answer}\n")

    # Get the answers from the file.
    return parser.parse("data/answers.txt")


def generate_entities(questions, answers):
    extractor = SpacyExtractor()
    result = []

    # Make tuples of questions and answers.
    for question, answer in zip(questions, answers):
        question_doc = extractor.extract_entities(question["text"])
        answer_doc = extractor.extract_entities(answer["text"])
        result.append(
            {
                "id": question["id"],
                "question": question["text"],
                "answer": answer["text"],
                "question_entities": [
                    {"text": ent.text, "type": ent.label_} for ent in question_doc.ents
                ],
                "answer_entities": [
                    {"text": ent.text, "type": ent.label_} for ent in answer_doc.ents
                ],
            }
        )

    return result


if __name__ == "__main__":
    parser = TxtParser()
    questions = parser.parse("data/input.txt")

    answers = generate_answers(questions, parser)
    entities = generate_entities(questions, answers)

    pprint(entities)
