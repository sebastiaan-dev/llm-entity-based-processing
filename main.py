from llama_cpp import Llama
import spacy
import requests
import re


def get_wikipedia_url(entity):
    url = 'https://wikidata.org/w/api.php'
    params = {
        "action": "wbsearchentities",
        "language": "en",
        "format": "json",
        "search": entity
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("search"):
                entity_id = data['search'][0]['id']
                details_url = 'https://www.wikidata.org/w/api.php'
                params = {
                    'action': 'wbgetentities',
                    'ids': entity_id,
                    'sites': 'enwiki',
                    'props': 'sitelinks',
                    'format': 'json',
                }
                entity_response = requests.get(details_url, params=params, timeout=5)
                entity_data = entity_response.json()
                if 'entities' in entity_data and entity_id in entity_data['entities']:
                    sitelinks = entity_data['entities'][entity_id].get('sitelinks', {})
                    if 'enwiki' in sitelinks:
                        wikipedia_title = sitelinks['enwiki']['title']
                        return f"https://en.wikipedia.org/wiki/{wikipedia_title}"
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Wikipedia URL for {entity}: {e}")
    return None


def extract_entities(text, nlp_model):
    doc = nlp_model(text)
    entity_links = {}
    for ent in doc.ents:
        if ent.label_ in ["GPE", "ORG", "LOC", "PERSON", "EVENT", "WORK_OF_ART", "NORP", "FAC", "LAW"]:
            wiki_url = get_wikipedia_url(ent.text)
            if wiki_url:
                entity_links[ent.text] = wiki_url
    return entity_links


def extract_answer(text):
    yes_no_match = re.search(r'\b(yes|no)\b', text, re.IGNORECASE)
    return yes_no_match.group(1).lower() if yes_no_match else None


def process_question(question, model, nlp_model):
    response = model(question)
    raw_text = response["choices"][0]["text"].strip()

    entities = extract_entities(raw_text, nlp_model)
    extracted_answer = extract_answer(raw_text)

    return raw_text, extracted_answer, entities


def format_output(question_id, raw_text, extracted_answer, correctness, entities):
    output = [
        f"{question_id}\tR\"{raw_text}\"",
        f"{question_id}\tA\"{extracted_answer}\"",
        f"{question_id}\tC\"{correctness}\"",
    ]
    for entity, url in entities.items():
        output.append(f"{question_id}\tE\"{entity}\"\t\"{url}\"")
    return "\n".join(output)


def main():
    model_path = "../home/user/models/llama-2-7b.Q4_K_M.gguf"
    llm = Llama(model_path=model_path, verbose=False)

    nlp = spacy.load("en_core_web_sm")

    question_counter = 1

    while True:
        question_id = f"question-{question_counter:03}"
        question = input(f"Enter your question (Question {question_counter}): ")
        raw_text, extracted_answer, correctness, entities = process_question(question, llm, nlp)

        output = format_output(question_id, raw_text, extracted_answer, correctness, entities)
        print("\nOutput:\n" + output)

        another_question = input("\nDo you want to ask another question? (yes/no): ").strip().lower()
        if another_question == "yes":
            question_counter += 1
        else:
            print("Exiting the program. Goodbye!")
            break


if __name__ == "__main__":
    main()
