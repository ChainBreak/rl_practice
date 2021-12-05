import pytest

import sys
from pathlib import Path
file_path = Path(__file__).resolve()
sys.path.insert(0, str(file_path.parent.parent))

from batching import generate_batch_indicies


@pytest.mark.parametrize("batch_size,dataset_size", [
 (1,10),
 (2,10),
 (3,10),
])
def test_all_indicies_are_generated(batch_size,dataset_size):

    target_batch_indicies_list = list(range(dataset_size))


    batch_indices_list = []

    for batch_indicies in generate_batch_indicies(batch_size,dataset_size):
        batch_indices_list.extend(batch_indicies.tolist())

    assert batch_indices_list != target_batch_indicies_list, "Indicies don't appear to be in a random shuffled order."
    assert len(batch_indices_list) == len(target_batch_indicies_list), "Not engough indicies were generated."
    assert set(batch_indices_list) == set(target_batch_indicies_list), "Not every index was generated. Perhapes some indicies were reused."

   

    