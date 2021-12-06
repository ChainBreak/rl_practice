import pytest

import sys
from pathlib import Path
file_path = Path(__file__).resolve()
sys.path.insert(0, str(file_path.parent.parent))

from plotting import generate_pendulum_state_space_mesh

@pytest.mark.parametrize("mesh_width,mesh_height", [
 (10,5),

])
def test_generate_pendulum_state_space_mesh(mesh_width,mesh_height):
    mesh = generate_pendulum_state_space_mesh(mesh_width,mesh_height)

    assert mesh.shape == [mesh_height,mesh_width,3]