import sys
from llama_cpp import Llama
import spacy
import requests

model_path = "models/llama-2-7b.Q4_K_M.gguf"
llm = Llama(model_path=model_path, verbose=False)
nlp = spacy.load("en_core_web_sm")


def extract_entities_spacy(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


def format_entity_output(question_id, entities):
    formatted_entities = []
    for entity, _ in entities:
        wiki_link = f"https://en.wikipedia.org/wiki/{entity.replace(' ', '_')}"
        formatted_entities.append(f'{question_id}<TAB>E"{entity}"<TAB>"{wiki_link}"')
    return formatted_entities


def extract_answer(question, response):
    lower_raw_text = response.lower()
    if "yes" in lower_raw_text:
        return "yes"
    elif "no" in lower_raw_text:
        return "no"

    doc = nlp(response)
    entities = [ent.text for ent in doc.ents]

    if entities:
        return entities[0]
    return "no"


def fact_check_answer(question, extracted_answer):
    if extracted_answer.lower() in ["yes", "no"]:
        return "correct"

    wiki_url = f"https://en.wikipedia.org/wiki/{extracted_answer.replace(' ', '_')}"
    response = requests.get(wiki_url)
    if response.status_code == 200:
        return "correct"
    return "incorrect"


def main():
    args = sys.argv[1].split("<TAB>")
    question_id = args[0]
    question = args[1]

    print(f"Asking the question '{question}' to {model_path} (this might take some time...)")

    output = llm(
        question,  # Prompt
        max_tokens=40,  # Generate up to 40 tokens
        echo=True  # Echo the prompt back in the output
    )
    response = output['choices'][0]["text"]

    print(f'{question_id}<TAB>R"{response.strip()}"')

    extracted_answer = extract_answer(question, response)
    print(f'{question_id}<TAB>A"{extracted_answer}"')

    correctness = fact_check_answer(question, extracted_answer)
    print(f'{question_id}<TAB>C"{correctness}"')

    entities = extract_entities_spacy(response)
    formatted_entities = format_entity_output(question_id, entities)
    for formatted_entity in formatted_entities:
        print(formatted_entity)


if __name__ == "__main__":
    main()
