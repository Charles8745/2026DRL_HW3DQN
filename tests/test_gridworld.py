import numpy as np
from src.gridworld_env import Gridworld


def test_static_initial_positions():
    """In static mode, Player=(0,3), Goal=(0,0), Pit=(0,1), Wall=(1,1)."""
    g = Gridworld(size=4, mode='static')
    assert g.board.components['Player'].pos == (0, 3)
    assert g.board.components['Goal'].pos == (0, 0)
    assert g.board.components['Pit'].pos == (0, 1)
    assert g.board.components['Wall'].pos == (1, 1)


def test_render_np_shape():
    """render_np returns (4 pieces, 4 rows, 4 cols) one-hot tensor."""
    g = Gridworld(size=4, mode='static')
    # Gridworld uses no masks; the mask path in render_np is not exercised here.
    arr = g.board.render_np()
    assert arr.shape == (4, 4, 4)
    assert arr.dtype == np.uint8
    # Each piece occupies exactly one cell
    for piece_layer in range(4):
        assert arr[piece_layer].sum() == 1


def test_make_move_left():
    """Moving left from (0,3) lands at (0,2)."""
    g = Gridworld(size=4, mode='static')
    g.makeMove('l')
    assert g.board.components['Player'].pos == (0, 2)


def test_reward_default():
    """Default reward (not on Goal/Pit) is -1."""
    g = Gridworld(size=4, mode='static')
    assert g.reward() == -1


def test_reward_goal():
    """Walking onto Goal yields +10. Static path: l, l, l from (0,3) to (0,0).
    But (0,1) is Pit and (0,0) is Goal — going l,l would step onto Pit.
    Direct test: place player on Goal artificially and check reward."""
    g = Gridworld(size=4, mode='static')
    g.board.components['Player'].pos = (0, 0)  # force onto Goal
    assert g.reward() == 10


def test_reward_pit():
    """Stepping onto Pit yields -10."""
    g = Gridworld(size=4, mode='static')
    g.board.components['Player'].pos = (0, 1)  # force onto Pit
    assert g.reward() == -10
