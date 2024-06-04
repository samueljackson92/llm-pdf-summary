import argparse
import json
import logging
from pathlib import Path
from pypdf import PdfReader
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO)


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def read_template(path: str) -> str:
    with Path(path).open() as handle:
        return handle.read()


def write_json(path: str, obj: object):
    with Path(path).open("w") as handle:
        json.dump(obj, handle, indent=4)


class MetaDataModel:
    def __init__(self, template) -> None:
        llm = ChatOllama(model="mistral")
        prompt = ChatPromptTemplate.from_template(template)
        self.chain = prompt | llm | JsonOutputParser()

    def create(self, text: str) -> dict:
        metadata = self.chain.invoke({"text": text[:3000]})
        return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("-o", "--output_folder", default="./out")
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)

    paths = input_folder.glob("**/*.pdf")
    paths = list(paths)

    output_folder.mkdir(exist_ok=True, parents=True)

    template = read_template("prompt.txt")
    model = MetaDataModel(template)

    n_files = len(paths)
    for i, path in enumerate(paths):
        logging.info(f"Creating metadata for {path}")
        text = read_pdf(Path(path).expanduser())
        metadata = model.create(text)
        output_file = output_folder / path.with_suffix(".json").name
        write_json(output_file, metadata)
        logging.info(
            f"Written metadata to {output_file}. Done {(i+1) / n_files*100:.2f}%"
        )


if __name__ == "__main__":
    main()
