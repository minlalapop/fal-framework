"""Non-IID client splitter."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from fal.dataset.client_splitters.client_splitter import ClientSplitter
from fal.dataset.client_splitters.splitter_utils import (
    item,
    num_samples,
    shuffle,
    split_contiguous_indices,
    take,
    validate_num_clients,
)


class NonIIDSplitter(ClientSplitter):
    """Label-skew split that groups labels before assigning samples to clients."""

    def __init__(
        self,
        shards_per_client: int = 2,
        shuffle_data: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.shards_per_client = shards_per_client
        self.shuffle = shuffle_data
        self.random_state = random_state

    def split(self, X: Any, y: Any, num_clients: int) -> list[tuple[Any, Any]]:
        validate_num_clients(num_clients)
        if self.shards_per_client <= 0:
            raise ValueError("`shards_per_client` must be a positive integer.")

        labels_to_indices: dict[Any, list[int]] = defaultdict(list)
        for index in range(num_samples(y)):
            labels_to_indices[item(y, index)].append(index)

        ordered_indices: list[int] = []
        for label in sorted(labels_to_indices, key=str):
            label_indices = labels_to_indices[label]
            if self.shuffle:
                shuffle(label_indices, self.random_state)
            ordered_indices.extend(label_indices)

        num_shards = max(num_clients * self.shards_per_client, num_clients)
        shards = [
            shard
            for shard in split_contiguous_indices(ordered_indices, num_shards)
            if shard
        ]

        if self.shuffle:
            shuffle(shards, self.random_state)

        client_indices = [[] for _ in range(num_clients)]
        for shard_index, shard in enumerate(shards):
            client_indices[shard_index % num_clients].extend(shard)

        return [(take(X, indices), take(y, indices)) for indices in client_indices]
