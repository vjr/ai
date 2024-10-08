import logging
import os
import sys

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse

from constants import (
    OPENAI_EMBEDDING_MODEL,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    PDF_FILENAME,
    PERSIST_DIR,
)

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

        parser = LlamaParse(result_type="text", show_progress=True, verbose=True)

        extractor = {".pdf": parser}

        reader = SimpleDirectoryReader(
            input_files=[PDF_FILENAME], file_extractor=extractor
        )

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
