from datasets import Dataset, DatasetDict



def generate_maze_example():
    # This dataset is not available online
    # The individual mazes need to be manually implemented according to section A.1 of the appendix, before we can proceed further.
    raise NotImplementedError


def build_maze_dataset(train_size: int = 1000, test_size: int = 100) -> DatasetDict:
    train_rows = [generate_maze_example(seed) for seed in range(42, 42 + train_size)]
    test_rows = [generate_maze_example(seed) for seed in range(4242, 4242 + test_size)]

    return DatasetDict({
        "train": Dataset.from_list(train_rows),
        "test": Dataset.from_list(test_rows),
    })