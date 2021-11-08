import pytest

import sys
from pathlib import Path
file_path = Path(__file__).resolve()
sys.path.insert(0, str(file_path.parent.parent))

from dataset import TransitionDataset


# @pytest.mark.parametrize("model_name", [
#  ("MfccNet"),
#  ("WaveNet"),
#  ("DenseMfccNet"),

# ])
def test_transition_dataset():
    


    data = [
            [
                ([1,2],0.1,[2,3],10),
                ([2,3],0.2,[3,4],20),
                ([3,4],0.3,[4,5],30),
            ],
            [
                ([1,2],0.1,[2,3],10),
                ([2,3],0.2,[3,4],20),
                ([3,4],0.3,[4,5],30),
            ]
    ]

    dataset = TransitionDataset(data)

    assert len(dataset) == 6

    assert dataset[0]["state"].tolist() == [1,2] 
    assert dataset[0]["action"].tolist() == pytest.approx(0.1)
    assert dataset[0]["next_state"].tolist() == [2,3] 
    assert dataset[0]["reward"].tolist() == [10]

    assert dataset[4]["state"].tolist() == [2,3] 
    assert dataset[4]["action"].tolist() == pytest.approx(0.2)
    assert dataset[4]["next_state"].tolist() == [3,4] 
    assert dataset[4]["reward"].tolist() == [20]
    