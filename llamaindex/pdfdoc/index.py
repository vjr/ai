import logging
import sys

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())


class Indexer:

    _system_prompt: str = """
    You are a MariaDB expert.

    You are expected to answer questions from the MariaDB documentation content provided.

    You must track all the tables of contents and the page numbers of the document.

    You must track the current chapter title and current chapter number for every page.
    """

    _index: VectorStoreIndex

    def __init__(self) -> None:

        _logger.info("Initialising...")

        Settings.llm = OpenAI(
            "gpt-4o", temperature=0.1, system_prompt=self._system_prompt
        )

        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    def index(self) -> None:

        _logger.info("Loading...")

        reader = SimpleDirectoryReader(input_files=["MariaDBServerKnowledgeBase.pdf"])

        documents = reader.load_data(show_progress=True)

        _logger.info("Indexing...")

        self._index = VectorStoreIndex.from_documents(documents, show_progress=True)

    def persist(self) -> None:

        _logger.info("Persisting...")
        self._index.storage_context.persist("storage")


def main() -> int:

    load_dotenv()

    indexer = Indexer()

    indexer.index()

    indexer.persist()

    return 0


if __name__ == "__main__":
    sys.exit(main())
