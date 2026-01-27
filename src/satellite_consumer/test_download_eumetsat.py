"""Tests for download_eumetsat module, specifically MD5 hash verification."""

import hashlib
import os
import tempfile

import pytest

from satellite_consumer.download_eumetsat import calculate_md5, verify_md5_hash


class TestCalculateMD5:
    """Tests for the calculate_md5 function."""

    def test_calculate_md5_returns_correct_hash(self) -> None:
        """Test that calculate_md5 returns the correct hash for a known file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_content = b"Hello, World!"
            f.write(test_content)
            f.flush()
            filepath = f.name

        try:
            expected_hash = hashlib.md5(test_content).hexdigest()  # noqa: S324
            actual_hash = calculate_md5(filepath)
            assert actual_hash == expected_hash
        finally:
            os.unlink(filepath)

    def test_calculate_md5_handles_large_file(self) -> None:
        """Test that calculate_md5 handles large files correctly."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            # Write 10MB of data
            test_content = b"x" * (10 * 1024 * 1024)
            f.write(test_content)
            f.flush()
            filepath = f.name

        try:
            expected_hash = hashlib.md5(test_content).hexdigest()  # noqa: S324
            actual_hash = calculate_md5(filepath)
            assert actual_hash == expected_hash
        finally:
            os.unlink(filepath)


class TestVerifyMD5Hash:
    """Tests for the verify_md5_hash function."""

    def test_verify_md5_hash_returns_true_for_matching_hash(self) -> None:
        """Test that verify_md5_hash returns True when hashes match."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_content = b"Test content for MD5"
            f.write(test_content)
            f.flush()
            filepath = f.name

        try:
            expected_hash = hashlib.md5(test_content).hexdigest()  # noqa: S324
            assert verify_md5_hash(filepath, expected_hash) is True
        finally:
            os.unlink(filepath)

    def test_verify_md5_hash_returns_false_for_mismatched_hash(self) -> None:
        """Test that verify_md5_hash returns False when hashes don't match."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"Some content")
            f.flush()
            filepath = f.name

        try:
            wrong_hash = "0" * 32  # Invalid hash
            assert verify_md5_hash(filepath, wrong_hash) is False
        finally:
            os.unlink(filepath)

    def test_verify_md5_hash_returns_true_for_empty_expected_hash(self) -> None:
        """Test that verify_md5_hash returns True when no expected hash is provided."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"Some content")
            f.flush()
            filepath = f.name

        try:
            assert verify_md5_hash(filepath, "") is True
            assert verify_md5_hash(filepath, None) is True  # type: ignore[arg-type]
        finally:
            os.unlink(filepath)

    def test_verify_md5_hash_is_case_insensitive(self) -> None:
        """Test that hash comparison is case-insensitive."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_content = b"Case insensitive test"
            f.write(test_content)
            f.flush()
            filepath = f.name

        try:
            expected_hash = hashlib.md5(test_content).hexdigest()  # noqa: S324
            assert verify_md5_hash(filepath, expected_hash.upper()) is True
            assert verify_md5_hash(filepath, expected_hash.lower()) is True
        finally:
            os.unlink(filepath)
