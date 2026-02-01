"""
Image handling utilities for ground truth evaluation.

Supports loading, encoding, and validating image files
for Visual QA and document extraction evaluation.
"""

import os
import base64
import struct
from typing import Optional, Tuple


def load_image(path: str) -> Optional[bytes]:
    """
    Load image file and return bytes.
    
    Args:
        path: Path to image file
    
    Returns:
        Image bytes or None
    """
    if not os.path.exists(path):
        return None
    
    with open(path, "rb") as f:
        return f.read()


def encode_image_base64(image: bytes) -> str:
    """
    Encode image as base64 string.
    
    Args:
        image: Image bytes
    
    Returns:
        Base64 encoded string
    """
    return base64.b64encode(image).decode('utf-8')


def decode_image_base64(encoded: str) -> bytes:
    """
    Decode base64 image string.
    
    Args:
        encoded: Base64 encoded string
    
    Returns:
        Image bytes
    """
    return base64.b64decode(encoded)


def validate_image(image: bytes) -> Tuple[bool, str]:
    """
    Validate image format and return format type.
    
    Args:
        image: Image bytes
    
    Returns:
        Tuple of (is_valid, format_name)
    """
    if len(image) < 8:
        return False, "unknown"
    
    # Check for JPEG
    if image[:2] == b'\xFF\xD8':
        return True, "jpeg"
    
    # Check for PNG
    if image[:8] == b'\x89PNG\r\n\x1a\n':
        return True, "png"
    
    # Check for GIF
    if image[:6] in (b'GIF87a', b'GIF89a'):
        return True, "gif"
    
    # Check for WebP
    if image[:4] == b'RIFF' and image[8:12] == b'WEBP':
        return True, "webp"
    
    # Check for BMP
    if image[:2] == b'BM':
        return True, "bmp"
    
    # Check for TIFF
    if image[:4] in (b'II\x2a\x00', b'MM\x00\x2a'):
        return True, "tiff"
    
    return False, "unknown"


def get_image_size(image: bytes) -> Tuple[int, int]:
    """
    Get image dimensions (width, height).
    
    Args:
        image: Image bytes
    
    Returns:
        Tuple of (width, height)
    """
    is_valid, format_name = validate_image(image)
    
    if not is_valid:
        return 0, 0
    
    if format_name == "png":
        return _get_png_size(image)
    elif format_name == "jpeg":
        return _get_jpeg_size(image)
    elif format_name == "gif":
        return _get_gif_size(image)
    elif format_name == "webp":
        return _get_webp_size(image)
    
    return 0, 0


def _get_png_size(image: bytes) -> Tuple[int, int]:
    """Get PNG image size."""
    try:
        if len(image) < 24:
            return 0, 0
        width = struct.unpack('>I', image[16:20])[0]
        height = struct.unpack('>I', image[20:24])[0]
        return width, height
    except Exception:
        return 0, 0


def _get_jpeg_size(image: bytes) -> Tuple[int, int]:
    """Get JPEG image size."""
    try:
        offset = 2
        while offset < len(image):
            if image[offset] != 0xFF:
                break
            
            marker = image[offset + 1]
            
            # SOF0, SOF2 markers contain dimensions
            if marker in (0xC0, 0xC2):
                height = struct.unpack('>H', image[offset + 5:offset + 7])[0]
                width = struct.unpack('>H', image[offset + 7:offset + 9])[0]
                return width, height
            
            # Skip to next marker
            if marker in (0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9):
                offset += 2
            else:
                length = struct.unpack('>H', image[offset + 2:offset + 4])[0]
                offset += 2 + length
        
        return 0, 0
    except Exception:
        return 0, 0


def _get_gif_size(image: bytes) -> Tuple[int, int]:
    """Get GIF image size."""
    try:
        if len(image) < 10:
            return 0, 0
        width = struct.unpack('<H', image[6:8])[0]
        height = struct.unpack('<H', image[8:10])[0]
        return width, height
    except Exception:
        return 0, 0


def _get_webp_size(image: bytes) -> Tuple[int, int]:
    """Get WebP image size."""
    try:
        if len(image) < 30:
            return 0, 0
        
        # VP8 format
        if image[12:16] == b'VP8 ':
            if image[23] != 0x9D or image[24] != 0x01 or image[25] != 0x2A:
                return 0, 0
            width = struct.unpack('<H', image[26:28])[0] & 0x3FFF
            height = struct.unpack('<H', image[28:30])[0] & 0x3FFF
            return width, height
        
        # VP8L format
        if image[12:16] == b'VP8L':
            bits = struct.unpack('<I', image[21:25])[0]
            width = (bits & 0x3FFF) + 1
            height = ((bits >> 14) & 0x3FFF) + 1
            return width, height
        
        return 0, 0
    except Exception:
        return 0, 0


def get_image_info(image: bytes) -> dict:
    """
    Get image file information.
    
    Args:
        image: Image bytes
    
    Returns:
        Dict with image info
    """
    is_valid, format_name = validate_image(image)
    width, height = get_image_size(image) if is_valid else (0, 0)
    
    return {
        "valid": is_valid,
        "format": format_name,
        "width": width,
        "height": height,
        "size_bytes": len(image),
    }


