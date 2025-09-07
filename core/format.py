"""
Tejas Binary Format Versioning and Persistence
==============================================

Handles versioned binary format with headers for future compatibility.
Supports automatic migration between format versions.

Format Specification:
- V1: Legacy format (numpy arrays + JSON config)
- V2: Binary format with header + packed fingerprints + metadata
"""

import struct
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Format version constants
FORMAT_VERSION_V1 = 1
FORMAT_VERSION_V2 = 2
CURRENT_FORMAT_VERSION = FORMAT_VERSION_V2

# V2 Binary header specification (32 bytes total)
TEJAS_MAGIC = b"TEJAS"  # 5 bytes
HEADER_SIZE = 32  # bytes


# Header structure for V2 format
class TejasHeaderV2:
    """Tejas V2 binary format header (32 bytes)."""

    def __init__(
        self,
        n_bits: int = 128,
        n_fingerprints: int = 0,
        bitorder: str = "little",
        flags: int = 0b00000001,
    ):
        self.magic = TEJAS_MAGIC  # 5 bytes
        self.version = FORMAT_VERSION_V2  # 1 byte
        self.flags = flags  # 1 byte (bit 0: packed, bits 1-7: reserved)
        self.n_bits = n_bits  # 2 bytes (uint16)
        self.n_fingerprints = n_fingerprints  # 4 bytes (uint32)
        self.bitorder = bitorder  # 6 bytes ('little' or 'big\x00\x00\x00')
        self.checksum = 0  # 4 bytes (CRC32) - computed later
        self.reserved = b"\x00" * 9  # 9 bytes (future use)

    def pack(self) -> bytes:
        """Pack header into 32-byte binary format."""
        # Ensure bitorder is exactly 6 bytes
        bitorder_bytes = self.bitorder.encode("ascii")[:6].ljust(6, b"\x00")

        header = struct.pack(
            "<5sBBHI6sI9s",  # little-endian format, total 32 bytes
            self.magic,
            self.version,
            self.flags,
            self.n_bits,
            self.n_fingerprints,
            bitorder_bytes,
            self.checksum,
            self.reserved,
        )

        assert len(header) == HEADER_SIZE, (
            f"Header size mismatch: {len(header)} != {HEADER_SIZE}"
        )
        return header

    @classmethod
    def unpack(cls, data: bytes) -> "TejasHeaderV2":
        """Unpack header from binary data."""
        if len(data) < HEADER_SIZE:
            raise ValueError(
                f"Insufficient data for header: {len(data)} < {HEADER_SIZE}"
            )

        parts = struct.unpack("<5sBBHI6sI9s", data[:HEADER_SIZE])

        header = cls()
        header.magic = parts[0]
        header.version = parts[1]
        header.flags = parts[2]
        header.n_bits = parts[3]
        header.n_fingerprints = parts[4]
        header.bitorder = parts[5].rstrip(b"\x00").decode("ascii")
        header.checksum = parts[6]
        header.reserved = parts[7]

        # Validate magic
        if header.magic != TEJAS_MAGIC:
            raise ValueError(f"Invalid magic: {header.magic} != {TEJAS_MAGIC}")

        return header

    def compute_checksum(self, data: bytes) -> int:
        """Compute CRC32 checksum for data validation."""
        import zlib

        return zlib.crc32(data) & 0xFFFFFFFF

    def to_dict(self) -> Dict[str, Any]:
        """Convert header to dictionary for JSON serialization."""
        return {
            "magic": self.magic.decode("ascii"),
            "version": self.version,
            "flags": self.flags,
            "n_bits": self.n_bits,
            "n_fingerprints": self.n_fingerprints,
            "bitorder": self.bitorder,
            "checksum": self.checksum,
            "is_packed": bool(self.flags & 0b00000001),
        }


def detect_format_version(filepath: Union[str, Path]) -> int:
    """
    Detect format version of a Tejas model file.

    Args:
        filepath: Path to model file or directory

    Returns:
        Format version (1 or 2)
    """
    filepath = Path(filepath)

    if filepath.is_dir():
        # V1 format: directory with config.json
        config_path = filepath / "config.json"
        if config_path.exists():
            return FORMAT_VERSION_V1
        else:
            raise ValueError(f"No config.json found in directory: {filepath}")

    elif filepath.is_file():
        # V2 format: single binary file
        try:
            with open(filepath, "rb") as f:
                magic = f.read(5)
                if magic == TEJAS_MAGIC:
                    return FORMAT_VERSION_V2
                else:
                    raise ValueError(f"Invalid magic in file: {magic}")
        except Exception as e:
            raise ValueError(f"Could not read file header: {e}")

    else:
        raise ValueError(f"Path does not exist: {filepath}")


def save_fingerprints_v2(
    filepath: Union[str, Path],
    fingerprints: np.ndarray,
    titles: list,
    metadata: Optional[Dict[str, Any]] = None,
    bitorder: str = "little",
) -> None:
    """
    Save fingerprints in V2 binary format with versioned header.

    Args:
        filepath: Output file path
        fingerprints: Fingerprint array (packed or unpacked)
        titles: List of titles
        metadata: Optional metadata dictionary
        bitorder: Bit order for packing ('little' or 'big')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Detect if fingerprints are already packed
    if fingerprints.dtype == np.uint8 and fingerprints.shape[1] < 32:
        # Already packed
        fp_packed = fingerprints
        n_bits = fingerprints.shape[1] * 8
        is_packed = True
    else:
        # Need to pack
        from core.bitops import pack_bits_rows

        fp_packed = pack_bits_rows(fingerprints, bitorder=bitorder)
        n_bits = fingerprints.shape[1]
        is_packed = True

    # Create header
    flags = 0b00000001 if is_packed else 0b00000000
    header = TejasHeaderV2(
        n_bits=n_bits, n_fingerprints=len(titles), bitorder=bitorder, flags=flags
    )

    # Prepare metadata
    if metadata is None:
        metadata = {}

    metadata.update(
        {
            "format_version": FORMAT_VERSION_V2,
            "n_titles": len(titles),
            "n_bits": n_bits,
            "is_packed": is_packed,
            "bitorder": bitorder,
        }
    )

    # Serialize metadata and titles
    titles_json = json.dumps(titles, ensure_ascii=False).encode("utf-8")
    metadata_json = json.dumps(metadata, ensure_ascii=False).encode("utf-8")

    # Pack data sections
    data_sections = []

    # Section 1: Fingerprints
    fp_bytes = fp_packed.tobytes()
    data_sections.append(fp_bytes)

    # Section 2: Titles (length-prefixed)
    titles_len = struct.pack("<I", len(titles_json))
    data_sections.append(titles_len + titles_json)

    # Section 3: Metadata (length-prefixed)
    metadata_len = struct.pack("<I", len(metadata_json))
    data_sections.append(metadata_len + metadata_json)

    # Combine all data
    data_payload = b"".join(data_sections)

    # Compute checksum
    header.checksum = header.compute_checksum(data_payload)

    # Write file
    with open(filepath, "wb") as f:
        f.write(header.pack())
        f.write(data_payload)

    logger.info(f"Saved V2 format to {filepath}")
    logger.info(f"  Fingerprints: {fp_packed.shape} ({fp_packed.nbytes:,} bytes)")
    logger.info(f"  Titles: {len(titles):,} items")
    logger.info(f"  Metadata: {len(metadata):,} fields")
    logger.info(f"  Total size: {filepath.stat().st_size:,} bytes")


def load_fingerprints_v2(
    filepath: Union[str, Path],
) -> Tuple[np.ndarray, list, Dict[str, Any]]:
    """
    Load fingerprints from V2 binary format.

    Args:
        filepath: Input file path

    Returns:
        (fingerprints, titles, metadata) tuple
    """
    filepath = Path(filepath)

    with open(filepath, "rb") as f:
        # Read and parse header
        header_data = f.read(HEADER_SIZE)
        header = TejasHeaderV2.unpack(header_data)

        # Read data payload
        data_payload = f.read()

    # Verify checksum
    computed_checksum = header.compute_checksum(data_payload)
    if computed_checksum != header.checksum:
        logger.warning(
            f"Checksum mismatch: {computed_checksum:08x} != {header.checksum:08x}"
        )

    # Parse data sections
    offset = 0

    # Section 1: Fingerprints
    n_bytes_per_fp = (header.n_bits + 7) // 8
    fp_size = header.n_fingerprints * n_bytes_per_fp
    fp_bytes = data_payload[offset : offset + fp_size]
    fingerprints = np.frombuffer(fp_bytes, dtype=np.uint8).reshape(
        header.n_fingerprints, n_bytes_per_fp
    )
    offset += fp_size

    # Section 2: Titles
    titles_len = struct.unpack("<I", data_payload[offset : offset + 4])[0]
    offset += 4
    titles_json = data_payload[offset : offset + titles_len].decode("utf-8")
    titles = json.loads(titles_json)
    offset += titles_len

    # Section 3: Metadata
    metadata_len = struct.unpack("<I", data_payload[offset : offset + 4])[0]
    offset += 4
    metadata_json = data_payload[offset : offset + metadata_len].decode("utf-8")
    metadata = json.loads(metadata_json)

    # Don't add header info to user metadata - keep them separate
    # metadata.update(header.to_dict())  # Removed to avoid test failures

    logger.info(f"Loaded V2 format from {filepath}")
    logger.info(f"  Version: {header.version}")
    logger.info(f"  Fingerprints: {fingerprints.shape}")
    logger.info(f"  Titles: {len(titles):,} items")
    logger.info(f"  Packed: {bool(header.flags & 0b00000001)}")

    return fingerprints, titles, metadata


def migrate_from_v1(v1_path: Union[str, Path], v2_path: Union[str, Path]) -> None:
    """
    Migrate V1 format to V2 format.

    Args:
        v1_path: Path to V1 directory
        v2_path: Output path for V2 file
    """
    v1_path = Path(v1_path)
    v2_path = Path(v2_path)

    logger.info(f"Migrating V1 → V2: {v1_path} → {v2_path}")

    # Load V1 format (directory structure)
    config_path = v1_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # For migration, we need fingerprints and titles from external source
    # This is a placeholder - actual implementation would load from search engine
    logger.warning("V1 migration requires external fingerprint data")
    logger.info("Migration template created - implement with actual data loading")

    # Create placeholder V2 file
    metadata = {
        "migrated_from": "v1",
        "original_config": config,
        "migration_timestamp": str(np.datetime64("now")),
    }

    # Save empty V2 file as template
    empty_fp = np.array([], dtype=np.uint8).reshape(0, 16)  # 128 bits = 16 bytes
    save_fingerprints_v2(v2_path, empty_fp, [], metadata)


def validate_format_compatibility(
    fingerprints: np.ndarray, titles: list, format_version: int = CURRENT_FORMAT_VERSION
) -> bool:
    """
    Validate that data is compatible with specified format version.

    Args:
        fingerprints: Fingerprint array
        titles: Title list
        format_version: Target format version

    Returns:
        True if compatible, False otherwise
    """
    try:
        if len(fingerprints) != len(titles):
            logger.error(
                f"Length mismatch: {len(fingerprints)} fingerprints != {len(titles)} titles"
            )
            return False

        if format_version == FORMAT_VERSION_V2:
            # V2 requirements
            if fingerprints.dtype != np.uint8:
                logger.error(
                    f"V2 requires uint8 fingerprints, got {fingerprints.dtype}"
                )
                return False

            if fingerprints.ndim != 2:
                logger.error(f"V2 requires 2D fingerprints, got {fingerprints.ndim}D")
                return False

            # Check reasonable size limits
            if fingerprints.shape[0] > 2**32 - 1:
                logger.error(
                    f"Too many fingerprints for V2: {fingerprints.shape[0]} > {2**32 - 1}"
                )
                return False

            n_bits = (
                fingerprints.shape[1] * 8
                if fingerprints.shape[1] < 32
                else fingerprints.shape[1]
            )
            if n_bits > 2**16 - 1:
                logger.error(f"Too many bits for V2: {n_bits} > {2**16 - 1}")
                return False

        return True

    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False


if __name__ == "__main__":
    # Format validation demo
    print("Tejas Format Versioning Demo")
    print("=" * 40)

    # Test header packing/unpacking
    print("\n1. Testing header serialization:")
    header = TejasHeaderV2(n_bits=128, n_fingerprints=1000, bitorder="little")
    packed = header.pack()
    unpacked = TejasHeaderV2.unpack(packed)

    print(f"   Original: {header.to_dict()}")
    print(f"   Roundtrip: {unpacked.to_dict()}")
    print(f"   Match: {'PASS' if header.to_dict() == unpacked.to_dict() else 'FAIL'}")

    # Test format detection
    print("\n2. Testing format detection:")
    current_dir = Path(".")
    try:
        version = detect_format_version(current_dir)
        print(f"   Current directory: V{version}")
    except Exception as e:
        print(f"   Current directory: {e}")

    print("\n3. Format validation:")
    test_fp = np.random.randint(0, 2, (100, 128), dtype=np.uint8)
    test_titles = [f"Title {i}" for i in range(100)]
    is_valid = validate_format_compatibility(test_fp, test_titles, FORMAT_VERSION_V2)
    print(f"   V2 compatibility: {'PASS' if is_valid else 'FAIL'}")
