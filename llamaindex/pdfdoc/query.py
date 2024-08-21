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
from llama_index.llms.openai import OpenAI

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_logger.addHandler(logging.StreamHandler())


class Queries:

    _system_prompt: str = """
    You are a MariaDB expert.
    You are expected to answer questions about the documentation content provided.
    At the end of your responses you must also provide additional info such as page number, section, etc.
    """

    _index: VectorStoreIndex

    _query_engine: BaseQueryEngine

    def __init__(self) -> None:

        Settings.llm = OpenAI(
            "gpt-4o", temperature=0.1, system_prompt=self._system_prompt
        )

    def load(self) -> None:

        storage = StorageContext.from_defaults(persist_dir="storage")

        _logger.info("Loading index")

        self._index = load_index_from_storage(storage)

        self._query_engine = self._index.as_query_engine()

    def query(self, query: str) -> str:

        _logger.info("Running query: %s", query)

        return self._query_engine.query(query)


def main() -> int:
    load_dotenv()

    queries = Queries()

    queries.load()

    response = queries.query("How do I create a table in MariaDB?")

    print("Response:", response)

    return 0


if __name__ == "__main__":
    sys.exit(main())
