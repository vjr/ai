import logging
import os

# import shutil
import sys

# import pymupdf
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex

# from llama_index.core.schema import Document, ImageDocument
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from constants import (
    OPENAI_EMBEDDING_MODEL,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    PDF_FILENAME,
    PERSIST_DIR,
)

# from pathlib import Path


# from llama_index.readers.file import PyMuPDFReader


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())


class Indexer:

    _system_prompt: str = """
    You are a MariaDB expert.

    You are expected to answer questions from the MariaDB documentation content provided.

    You must track the main table of contents and the page numbers of the document.

    You must track the current chapter for every page from the main table of contents.
    """

    _index: VectorStoreIndex

    def __init__(self) -> None:

        _logger.info("Initialising...")

        Settings.llm = OpenAI(
            OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            system_prompt=self._system_prompt,
        )

        Settings.embed_model = OpenAIEmbedding(model=OPENAI_EMBEDDING_MODEL)

    def index(self) -> None:

        _logger.info(f"Loading {PDF_FILENAME} ...")

        # pdfreader = PyMuPDFReader()
        # pages = pdfreader.load_data(file_path=Path(PDF_FILENAME))
        # documents = []
        # for page in pages:
        #     document = ImageDocument()
        #     documents.append(page.get_p)

        # if os.path.exists(PERSIST_DIR):
        #     shutil.rmtree(PERSIST_DIR)

        # os.makedirs(f"{PERSIST_DIR}/images", exist_ok=True)

        # doc = pymupdf.open(PDF_FILENAME)

        # for page in doc.pages():
        #     pixmap = page.get_pixmap()
        #     pixmap.save(f"{PERSIST_DIR}/images/page-{page.number}.png")

        # extractor = {".pdf": pdfreader}

        # reader = SimpleDirectoryReader(
        #     input_files=[PDF_FILENAME], file_extractor=extractor
        # )

        # reader = SimpleDirectoryReader(input_dir=f"{PERSIST_DIR}/images")

        reader = SimpleDirectoryReader(input_files=[PDF_FILENAME])

        documents = reader.load_data(show_progress=True)

        _logger.info("Indexing...")

        self._index = VectorStoreIndex.from_documents(documents, show_progress=True)

    def persist(self) -> None:

        _logger.info("Persisting...")
        self._index.storage_context.persist(PERSIST_DIR)


def main() -> int:

    if os.path.exists(PERSIST_DIR):

        choice = input(
            f"Index persist directory '{PERSIST_DIR}' already exists. Overwrite? [y/n]: "
        ).lower()

        if choice != "y" and choice != "yes":
            _logger.info("Exiting...")
            return 0
        else:
            _logger.info("Overwriting...")

    load_dotenv()

    indexer = Indexer()

    indexer.index()

    indexer.persist()

    return 0


if __name__ == "__main__":
    sys.exit(main())
