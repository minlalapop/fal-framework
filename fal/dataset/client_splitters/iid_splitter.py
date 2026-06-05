"""IID client splitter."""

from __future__ import annotations

from typing import Any

from fal.dataset.client_splitters.client_splitter import ClientSplitter
from fal.dataset.client_splitters.splitter_utils import (
    num_samples,
    shuffle,
    split_indices,
    take,
    validate_num_clients,
)


class IIDSplitter(ClientSplitter):
    """Randomly distribute samples across clients."""

    def __init__(self, shuffle_data: bool = True, random_state: int | None = None) -> None:
        self.shuffle = shuffle_data
        self.random_state = random_state

    def split(self, X: Any, y: Any, num_clients: int) -> list[tuple[Any, Any]]:
        validate_num_clients(num_clients)
        indices = list(range(num_samples(y)))

        if self.shuffle:
            shuffle(indices, self.random_state)

        chunks = split_indices(indices, num_clients)
        return [(take(X, chunk), take(y, chunk)) for chunk in chunks]
