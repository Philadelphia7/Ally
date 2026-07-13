from app.azure_clients import iter_embedding_batches


def test_iter_embedding_batches_respects_item_and_character_limits():
    texts = ["a" * 5, "b" * 5, "c" * 9, "d" * 2]

    batches = list(
        iter_embedding_batches(
            texts,
            max_batch_items=2,
            max_batch_characters=10,
        )
    )

    assert batches == [["a" * 5, "b" * 5], ["c" * 9], ["d" * 2]]


def test_iter_embedding_batches_keeps_oversized_single_text():
    batches = list(
        iter_embedding_batches(
            ["x" * 25, "y"],
            max_batch_items=10,
            max_batch_characters=10,
        )
    )

    assert batches == [["x" * 25], ["y"]]

