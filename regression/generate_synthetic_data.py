from datetime import datetime
import json
import os
from alive_progress import alive_bar
from ollama import ChatResponse, chat


TARGET_ENTITY_TYPES = [
    "CARDINAL",
    "DATE",
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "MONEY",
    "NORP",
    "ORDINAL",
    "ORG",
    "PERCENT",
    "PERSON",
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "WORK_OF_ART",
]
ENTITY_LIST = ",".join(TARGET_ENTITY_TYPES)
ROOT_TOPICS = [
    "Technology",
    "Science",
    "Health & Medicine",
    "History",
    "Geography",
    "Environment",
    "Education",
    "Art & Culture",
    "Business & Economics",
    "Politics & Government",
    "Law & Justice",
    "Sports & Recreation",
    "Travel & Tourism",
    "Food & Cuisine",
    "Philosophy & Ethics",
    "Religion & Spirituality",
    "Literature & Language",
    "Entertainment & Media",
    "Music & Performing Arts",
    "Mathematics",
    "Physics",
    "Chemistry",
    "Biology",
    "Engineering",
    "Astronomy & Space",
    "Psychology",
    "Sociology",
    "Anthropology",
    "Economics",
    "Finance & Investment",
    "Marketing & Advertising",
    "Cybersecurity",
    "Artificial Intelligence",
    "Software Development",
    "Hardware & Electronics",
    "Social Media & Communication",
    "Transportation & Logistics",
    "Military & Defense",
    "Architecture & Urban Planning",
    "Fashion & Design",
    "Mythology & Folklore",
    "Archaeology",
    "Energy & Sustainability",
    "Agriculture & Farming",
    "Animals & Wildlife",
    "Oceanography & Marine Science",
    "Space Exploration",
    "Historical Events",
    "Global Conflicts & Wars",
    "Cultural Heritage",
    "World Religions",
    "Climate Change & Global Warming",
    "Human Rights & Social Justice",
    "Public Policy & Administration",
    "International Relations",
    "Disaster Management",
    "Artificial Life & Robotics",
]
MAX_DEPTH = 2
SUBTOPIC_COUNT = 3
QUESTION_COUNT = 20
OUTPUT_DIR = "data_synthetic"


def query_ollama(query: str) -> str:
    response: ChatResponse = chat(
        model="qwen2.5:32b",
        messages=[
            {
                "role": "user",
                "content": query,
            },
        ],
    )
    return response["message"]["content"]


def get_subtopics(topic: str) -> list[str]:
    prompt = (
        f"You are an expert in categorization. Divide the topic '{topic}' into "
        f"{SUBTOPIC_COUNT} subtopics. Provide the subtopics as a comma-separated list. Keep topics English."
    )

    response = query_ollama(prompt)
    arr = response.strip().split(",")

    return [x.strip() for x in arr]


def generate_questions(topic: str, entity_target: str) -> list[str]:
    prompt = (
        f"Generate {QUESTION_COUNT} questions related to the topic '{topic}', which target entity type {entity_target}. "
        f"Each question should be followed by its entity type, separated by a comma. "
        f"Format each line as follows:\n"
        f"<question>,<entity-type>"
        f"If you can't generate more questions, stop. Use English."
    )

    response = query_ollama(prompt)
    arr = response.strip().split("\n")

    return [x.strip() for x in arr]


def generate_questions_nil(topic: str) -> list[str]:
    prompt = (
        f"Generate {QUESTION_COUNT} questions related to the topic '{topic}', which target entity type NIL. "
        f"The other are: {ENTITY_LIST}"
        f"Each question should be followed by its entity type, separated by a comma. "
        f"Format each line as follows:\n"
        f"<question>,<entity-type>"
        f"If you can't generate more questions, stop. Use English."
    )

    response = query_ollama(prompt)
    arr = response.strip().split("\n")

    return [x.strip() for x in arr]


def save_topic_tree(tree: dict, file_name: str):
    with open(file_name, "w") as file:
        json.dump(tree, file, indent=4, ensure_ascii=False)


def save_questions_to_csv(topic: str, questions: list[str]):
    """
    Saves generated questions for a topic into a CSV file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    file_name = os.path.join(
        OUTPUT_DIR, f"{topic.replace(' ', '_').lower()}_{timestamp}.csv"
    )

    # Skip if file already exists
    if os.path.exists(file_name):
        print(f"File already exists for {topic}, skipping...")
        return

    with open(file_name, "w", newline="", encoding="utf-8") as file:
        file.write("Question,Entity-Type\n")
        file.writelines(f"{question}\n" for question in questions)

    print(f"Saved questions for {topic} to {file_name}.")


def generate_topic_tree(
    tree: dict, topics: list[str], depth: int = 0
) -> dict[str, list[str]]:
    print(f"Generating topic tree for {topics} at depth {depth}")

    if depth > MAX_DEPTH:
        return

    for topic in topics:
        if depth < MAX_DEPTH:
            subtopics = get_subtopics(topic)

            tree[topic] = {"subtopics": subtopics}
            generate_topic_tree(tree[topic], subtopics, depth + 1)
        else:
            tree[topic] = {"questions": []}

            with alive_bar(len(TARGET_ENTITY_TYPES)) as bar:
                for entity_target in TARGET_ENTITY_TYPES:
                    questions = generate_questions(topic, entity_target)
                    tree[topic]["questions"].extend(questions)
                    bar()

            questions_nil = generate_questions_nil(topic)
            tree[topic]["questions"].extend(questions_nil)

            save_questions_to_csv(topic, tree[topic]["questions"])


def main():
    tree = {}
    generate_topic_tree(tree, ROOT_TOPICS)
    save_topic_tree(tree, "topic_tree.json")


if __name__ == "__main__":
    main()
