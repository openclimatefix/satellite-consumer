"""Renaming variables for the destination store before writing."""

import json

import numpy as np
import xarray as xr

from satellite_consumer import storage


def test_variables_are_renamed_from_the_store_file(tmp_path, monkeypatch):
    names = tmp_path / "names.json"
    names.write_text(json.dumps({"IR_108": "channel_09_brightness_temperature_10_8um"}))
    monkeypatch.setenv("SATCONS_RENAME_JSON", str(names))
    ds = xr.Dataset({"IR_108": (("time", "y"), np.zeros((1, 2)))})
    encoding = {"_ARRAY_DIMENSIONS": ["time", "y"], "IR_108": {"dtype": "float32"}}
    renamed, renamed_encoding = storage._rename_for_store(ds, encoding)
    assert "channel_09_brightness_temperature_10_8um" in renamed
    assert renamed["channel_09_brightness_temperature_10_8um"].attrs["source_name"] == "IR_108"
    assert "channel_09_brightness_temperature_10_8um" in renamed_encoding


def test_nothing_changes_without_a_store_file(monkeypatch):
    monkeypatch.delenv("SATCONS_RENAME_JSON", raising=False)
    ds = xr.Dataset({"IR_108": (("time",), np.zeros(1))})
    assert storage._rename_for_store(ds, {"_ARRAY_DIMENSIONS": ["time"]})[0] is ds
