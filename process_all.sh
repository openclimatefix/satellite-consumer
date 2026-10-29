#!/usr/bin/env bash
# One command per line: every MTG / MSG / IODC / Himawari satellite-resolution pair.
#
# Resolutions come from the native channel resolutions in cmd/application.conf:
#   mtg (odegree-12)             -> 1000m (vis/nir), 2000m (ir/wv)
#   mtg-highres (odegree-12-hr)  ->  500m (vis_06, nir_22), 1000m (ir_38, ir_105)
#   seviri (odegree/iodc)        -> 1000m (HRV), 3000m (everything else)
#   ahi (himawari)               ->  500m (B03), 1000m (B01/B02/B04), 2000m (rest)
#
# Credentials are read from the environment, so export these before running:
#   EUMETSAT_CONSUMER_KEY EUMETSAT_CONSUMER_SECRET AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
# Each line overrides the time window with SATCONS_START_TIMESTAMP / SATCONS_END_TIMESTAMP if set.
# Run a single line directly, or feed the file to a runner, e.g.:
#   grep -v '^\s*#' process_all.sh | grep . | parallel -j2

# MTG (Meteosat-12, FCI)
SATCONS_SATELLITE="odegree-12-highres" SATCONS_RESOLUTION="500" SATCONS_START_TIMESTAMP="${SATCONS_START_TIMESTAMP:-2026-01-01T00:00:00Z}" SATCONS_END_TIMESTAMP="${SATCONS_END_TIMESTAMP:-2026-10-01T00:00:00Z}" SATCONS_RAW_PATH="." SATCONS_ZARR_PATH="s3://us-west-2.opendata.source.coop/bkr/geo/mtg_500m.icechunk" SATCONS_ICECHUNK=True SATCONS_LOW_MEMORY=true pixi run sat-consumer
SATCONS_SATELLITE="odegree-12-highres" SATCONS_RESOLUTION="1000" SATCONS_START_TIMESTAMP="${SATCONS_START_TIMESTAMP:-2026-01-01T00:00:00Z}" SATCONS_END_TIMESTAMP="${SATCONS_END_TIMESTAMP:-2026-10-01T00:00:00Z}" SATCONS_RAW_PATH="." SATCONS_ZARR_PATH="s3://us-west-2.opendata.source.coop/bkr/geo/mtg_highres_1000m.icechunk" SATCONS_ICECHUNK=True pixi run sat-consumer
SATCONS_SATELLITE="odegree-12" SATCONS_RESOLUTION="1000" SATCONS_START_TIMESTAMP="${SATCONS_START_TIMESTAMP:-2026-01-01T00:00:00Z}" SATCONS_END_TIMESTAMP="${SATCONS_END_TIMESTAMP:-2026-10-01T00:00:00Z}" SATCONS_RAW_PATH="." SATCONS_ZARR_PATH="s3://us-west-2.opendata.source.coop/bkr/geo/mtg_1000m.icechunk" SATCONS_ICECHUNK=True pixi run sat-consumer
SATCONS_SATELLITE="odegree-12" SATCONS_RESOLUTION="2000" SATCONS_START_TIMESTAMP="${SATCONS_START_TIMESTAMP:-2026-01-01T00:00:00Z}" SATCONS_END_TIMESTAMP="${SATCONS_END_TIMESTAMP:-2026-10-01T00:00:00Z}" SATCONS_RAW_PATH="." SATCONS_ZARR_PATH="s3://us-west-2.opendata.source.coop/bkr/geo/mtg_2000m.icechunk" SATCONS_ICECHUNK=True pixi run sat-consumer

# MSG (Meteosat 0 degree, SEVIRI)
SATCONS_SATELLITE="odegree" SATCONS_RESOLUTION="1000" SATCONS_START_TIMESTAMP="${SATCONS_START_TIMESTAMP:-2025-01-01T00:00:00Z}" SATCONS_END_TIMESTAMP="${SATCONS_END_TIMESTAMP:-2026-10-01T00:00:00Z}" SATCONS_RAW_PATH="." SATCONS_ZARR_PATH="s3://us-west-2.opendata.source.coop/bkr/geo/msg_1000m.icechunk" SATCONS_ICECHUNK=True pixi run sat-consumer
SATCONS_SATELLITE="odegree" SATCONS_RESOLUTION="3000" SATCONS_START_TIMESTAMP="${SATCONS_START_TIMESTAMP:-2025-01-01T00:00:00Z}" SATCONS_END_TIMESTAMP="${SATCONS_END_TIMESTAMP:-2026-10-01T00:00:00Z}" SATCONS_RAW_PATH="." SATCONS_ZARR_PATH="s3://us-west-2.opendata.source.coop/bkr/geo/msg_3000m.icechunk" SATCONS_ICECHUNK=True pixi run sat-consumer

# IODC (Meteosat Indian Ocean, SEVIRI)
SATCONS_SATELLITE="iodc" SATCONS_RESOLUTION="1000" SATCONS_START_TIMESTAMP="${SATCONS_START_TIMESTAMP:-2025-01-01T00:00:00Z}" SATCONS_END_TIMESTAMP="${SATCONS_END_TIMESTAMP:-2026-10-01T00:00:00Z}" SATCONS_RAW_PATH="." SATCONS_ZARR_PATH="s3://us-west-2.opendata.source.coop/bkr/geo/iodc_1000m_test.icechunk" SATCONS_ICECHUNK=True pixi run sat-consumer
SATCONS_SATELLITE="iodc" SATCONS_RESOLUTION="3000" SATCONS_START_TIMESTAMP="${SATCONS_START_TIMESTAMP:-2025-01-01T00:00:00Z}" SATCONS_END_TIMESTAMP="${SATCONS_END_TIMESTAMP:-2026-10-01T00:00:00Z}" SATCONS_RAW_PATH="." SATCONS_ZARR_PATH="s3://us-west-2.opendata.source.coop/bkr/geo/iodc_3000m_test.icechunk" SATCONS_ICECHUNK=True pixi run sat-consumer

# Himawari (AHI)
SATCONS_SATELLITE="himawari" SATCONS_RESOLUTION="500" SATCONS_START_TIMESTAMP="${SATCONS_START_TIMESTAMP:-2025-01-01T00:00:00Z}" SATCONS_END_TIMESTAMP="${SATCONS_END_TIMESTAMP:-2026-10-01T00:00:00Z}" SATCONS_RAW_PATH="." SATCONS_ZARR_PATH="s3://us-west-2.opendata.source.coop/bkr/geo/himawari_500m_test.icechunk" SATCONS_ICECHUNK=True SATCONS_LOW_MEMORY=true pixi run sat-consumer
SATCONS_SATELLITE="himawari" SATCONS_RESOLUTION="1000" SATCONS_START_TIMESTAMP="${SATCONS_START_TIMESTAMP:-2025-01-01T00:00:00Z}" SATCONS_END_TIMESTAMP="${SATCONS_END_TIMESTAMP:-2026-10-01T00:00:00Z}" SATCONS_RAW_PATH="." SATCONS_ZARR_PATH="s3://us-west-2.opendata.source.coop/bkr/geo/himawari_1000m_test.icechunk" SATCONS_ICECHUNK=True pixi run sat-consumer
SATCONS_SATELLITE="himawari" SATCONS_RESOLUTION="2000" SATCONS_START_TIMESTAMP="${SATCONS_START_TIMESTAMP:-2025-01-01T00:00:00Z}" SATCONS_END_TIMESTAMP="${SATCONS_END_TIMESTAMP:-2026-10-01T00:00:00Z}" SATCONS_RAW_PATH="." SATCONS_ZARR_PATH="s3://us-west-2.opendata.source.coop/bkr/geo/himawari_2000m_test.icechunk" SATCONS_ICECHUNK=True pixi run sat-consumer
