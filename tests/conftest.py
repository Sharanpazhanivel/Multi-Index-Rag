"""Pytest fixtures: mock retrievers, sample queries."""
import pytest


@pytest.fixture
def sample_query() -> str:
    return "What is our refund policy?"


@pytest.fixture
def sample_sql_query() -> str:
    return "SELECT * FROM orders WHERE status = 'pending'"
