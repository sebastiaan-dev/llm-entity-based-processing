import os
import re

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
INPUT_DIR = "data_synthetic"
OUTPUT_DIR = "data_synthetic"
OUTPUT_FILE = "data_synthetic.csv"

chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]+")


def contains_chinese(sentence):
    return bool(chinese_char_pattern.search(sentence))


def merge_csv_files():
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

    lines = []
    filtered_count = 0

    for file in csv_files:
        with open(os.path.join(INPUT_DIR, file), "r") as f:
            file_lines = f.readlines()[1:]
            for line in file_lines:
                if len(line.rsplit(",", 1)) == 1:
                    filtered_count += 1
                    continue
                if contains_chinese(line):
                    filtered_count += 1
                    continue

                # Handle sentences with commas in the Question field
                question, entity_type = line.rsplit(",", 1)

                # Skip entities which do not conform to the target entity types
                if not entity_type.strip() in TARGET_ENTITY_TYPES:
                    filtered_count += 1
                    continue

                question = question.strip()
                # Escape double quotes as these are used as delimiters in CSV
                question = question.replace('"', '""')

                # Wrap Question in quotes if it contains commas or newlines
                if "," in question or "\n" in question:
                    question = f'"{question}"'

                lines.append(f"{question},{entity_type}")

    with open(os.path.join(OUTPUT_DIR, OUTPUT_FILE), "w") as f:
        f.write("Question,Entity-Type\n")
        f.writelines(lines)

    print(f"Filtered {filtered_count} lines.")


if __name__ == "__main__":
    merge_csv_files()
