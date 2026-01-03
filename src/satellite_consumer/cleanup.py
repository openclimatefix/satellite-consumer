"""Script to clean up duplicate timestamps in existing Zarr stores.

This script identifies and removes duplicate timestamps from satellite data stores,
keeping the data from the satellite with the orbital position closest to the
prime meridian (0°). This is useful for cleaning up stores that were affected
by satellite orbital transitions (e.g., Meteosat moving from 9.5° to 0°).

Usage:
    # Preview what would be removed (dry run)
    sat-cleanup /path/to/store.zarr --dry-run

    # Actually remove duplicates
    sat-cleanup /path/to/store.zarr

    # For S3 stores
    sat-cleanup s3://bucket/path/store.zarr

    # For icechunk repositories
    sat-cleanup s3://bucket/path/store.zarr --use-icechunk
"""

import argparse
import logging
import sys

from satellite_consumer import storage

log = logging.getLogger(__name__)


def main() -> int:
    """Clean up duplicate timestamps from a Zarr store.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Remove duplicate timestamps from satellite Zarr stores. "
        "Keeps data from satellites closest to the prime meridian (0°).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview duplicates without removing
  sat-cleanup /path/to/store.zarr --dry-run

  # Remove duplicates from a local store
  sat-cleanup /path/to/store.zarr

  # Remove duplicates from an S3 store
  sat-cleanup s3://bucket/satellite-data.zarr

  # Remove duplicates from an icechunk repository
  sat-cleanup s3://bucket/icechunk-repo --use-icechunk
        """,
    )

    parser.add_argument(
        "zarr_path",
        help="Path to the Zarr store to clean (local path or s3://... or gcs://...)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually modifying the store",
    )
    parser.add_argument(
        "--time-dim",
        default="time",
        help="Name of the time dimension coordinate (default: 'time')",
    )
    parser.add_argument(
        "--use-icechunk",
        action="store_true",
        help="Treat the path as an icechunk repository instead of a regular Zarr store",
    )
    parser.add_argument(
        "--aws-access-key-id",
        default=None,
        help="AWS access key ID for S3 access",
    )
    parser.add_argument(
        "--aws-secret-access-key",
        default=None,
        help="AWS secret access key for S3 access",
    )
    parser.add_argument(
        "--aws-region",
        default=None,
        help="AWS region name for S3 access",
    )
    parser.add_argument(
        "--aws-endpoint-url",
        default=None,
        help="AWS endpoint URL for S3 access (for S3-compatible services)",
    )
    parser.add_argument(
        "--gcs-token",
        default=None,
        help="GCS token for Google Cloud Storage access",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (debug) logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Get the destination (either icechunk repo or path string)
        if args.use_icechunk:
            dst = storage.get_icechunk_repo(
                path=args.zarr_path,
                aws_access_key_id=args.aws_access_key_id,
                aws_secret_access_key=args.aws_secret_access_key,
                aws_region_name=args.aws_region,
                aws_endpoint_url=args.aws_endpoint_url,
                gcs_token=args.gcs_token,
            )
        else:
            dst = args.zarr_path

        # Run duplicate removal
        removed = storage.remove_duplicate_times(
            dst=dst,
            time_dim=args.time_dim,
            dry_run=args.dry_run,
        )

        # Print summary
        if args.dry_run:
            print(f"\n[DRY RUN] Would remove {len(removed)} duplicate timestamp(s)")
            if removed:
                print("Timestamps that would be removed:")
                for ts in removed:
                    print(f"  - {ts}")
        else:
            print(f"\nRemoved {len(removed)} duplicate timestamp(s)")
            if removed:
                print("Removed timestamps:")
                for ts in removed:
                    print(f"  - {ts}")

        return 0

    except FileNotFoundError as e:
        log.error("Store not found: %s", e)
        return 1
    except Exception as e:
        log.exception("Error during cleanup: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
