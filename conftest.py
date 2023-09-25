"""Pytest fixtures for doctests."""
import pytest

import pysysid


@pytest.fixture(autouse=True)
def add_pysysid(doctest_namespace):
    """Add ``pysysid`` to namespace."""
    doctest_namespace['pysysid'] = pysysid
