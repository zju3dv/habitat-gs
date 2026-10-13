import hashlib

import pytest

from scripts_gs.benchmark.verify import verify_files


def test_detects_same_length_corruption_and_path_escape(tmp_path):
    data = b"original"
    path = tmp_path / "asset"
    path.write_bytes(data)
    files = {"asset": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}
    assert verify_files(tmp_path, files) == 1
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="mismatch"):
        verify_files(tmp_path, files)
    with pytest.raises(ValueError, match="unsafe"):
        verify_files(tmp_path, {"../outside": files["asset"]})
