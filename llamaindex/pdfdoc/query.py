import logging
import sys

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from constants import (
    OPENAI_EMBEDDING_MODEL,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
    PERSIST_DIR,
)

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())


class Queries:

    _system_prompt: str = """
    You are a MariaDB expert.

    You are expected to answer questions from the MariaDB documentation content provided.

    Append your response with the chapter, number, page number and total pages.

    Do not format your response in any way. Only output plain text.
    """

    _index: VectorStoreIndex

    _query_engine: BaseQueryEngine

    def __init__(self) -> None:

        _logger.info("Initialising...")

        Settings.llm = OpenAI(
            OPENAI_MODEL,
            temperature=OPENAI_TEMPERATURE,
            system_prompt=self._system_prompt,
        )

        Settings.embed_model = OpenAIEmbedding(model=OPENAI_EMBEDDING_MODEL)

    def load(self) -> None:

        _logger.info("Loading index...")

        try:
            storage = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        except FileNotFoundError:
            _logger.error(
                "No persisted index found. Please run 'python index.py' first."
            )
            sys.exit(1)

        _logger.info("Creating index...")

        self._index = load_index_from_storage(storage)

        self._query_engine = self._index.as_query_engine()

    def query(self, query: str) -> str:

        _logger.info("Thinking...")

        return self._query_engine.query(query)


def main() -> int:

    load_dotenv()

    queries = Queries()

    queries.load()

    query = input("Ask: ")

    while query not in ["exit", "quit", "bye"]:
        answer = queries.query(query)
        print("Answer:", answer)
        query = input("Ask: ")

    return 0


if __name__ == "__main__":
    sys.exit(main())
