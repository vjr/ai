import logging
import sys

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())


class Indexer:

    _index: VectorStoreIndex

    def index(self) -> None:

        documents = SimpleDirectoryReader(
            input_files=["MariaDBServerKnowledgeBase.pdf"]
        ).load_data(show_progress=True)

        _logger.info("Indexing...")
        self._index = VectorStoreIndex.from_documents(documents)

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
