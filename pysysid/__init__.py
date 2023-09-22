"""System identification library in Python."""

from .classical import Arx
from .util import (
    block_hankel,
    combine_episodes,
    extract_initial_conditions,
    extract_input,
    extract_output,
    split_episodes,
    strip_initial_conditions,
)
