from abc import ABC, abstractmethod


class Parser(ABC):
    def read(self, file_path: str) -> str:
        """
        Read the contents of a file and return it as a string.

        :param file_path: str
        :return: str
        """
        with open(file_path, "r") as file:
            return file.read()

    @abstractmethod
    def parse(self, data):
        pass


class TxtParser(Parser):
    def parse(self, file_path: str) -> list:
        """
        Parse incoming data of the format:

            <question-id><TAB><question-text><newline>

        Into a list of dictionaries with the following format:

            {
                "id": <question-id>,
                "text": <question-text>
            }

        :param data: str
        :return: list of dict
        """
        lines = self.read(file_path).split("\n")
        result = []

        for line in lines:
            if not line:
                continue
            try:
                id, text = line.split(maxsplit=1)
                result.append({"id": id, "text": text})
            except ValueError:
                raise ValueError(f"Invalid line format: {line}")

        return result
