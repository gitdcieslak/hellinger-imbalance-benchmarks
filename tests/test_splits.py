import numpy as np

from hib.splits import generate_stratified_repeated_splits


def test_split_generator_creates_five_stratified_half_splits():
    y = np.array([0] * 100 + [1] * 10)

    splits = generate_stratified_repeated_splits(y, n_repeats=5, test_size=0.5, random_seed=7)

    assert len(splits) == 5
    assert [split.repeat_id for split in splits] == [0, 1, 2, 3, 4]
    assert [split.split_seed for split in splits] == [7, 8, 9, 10, 11]
    assert all(split.train_idx.size == 55 for split in splits)
    assert all(split.test_idx.size == 55 for split in splits)


def test_each_split_preserves_both_classes_when_feasible():
    y = np.array([0] * 200 + [1] * 20)

    splits = generate_stratified_repeated_splits(y, n_repeats=5, test_size=0.5, random_seed=0)

    for split in splits:
        train_y = y[split.train_idx]
        test_y = y[split.test_idx]
        assert np.unique(train_y).tolist() == [0, 1]
        assert np.unique(test_y).tolist() == [0, 1]
