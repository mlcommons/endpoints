import pytest
from inference_endpoint.utils import byte_quantity_to_str


@pytest.mark.parametrize(
    "n_bytes, max_unit, expected",
    [
        (1024, "GB", "1KB"),
        (1024 * 1024, "TB", "1MB"),
        (1024 * 1024, "GB", "1MB"),
        (1024 * 1024 * 1024, "GB", "1GB"),
        (1024 * 1024 * 1024 * 1024, "GB", "1024GB"),
        (1024 * 1024 * 1024 * 1024, "TB", "1TB"),
        (5 * 1024 * 1024, "TB", "5MB"),
        (1024 * 1024, "KB", "1024KB"),
    ],
)
def test_byte_quantity_to_str(n_bytes, max_unit, expected):
    assert byte_quantity_to_str(n_bytes, max_unit=max_unit) == expected
