"""Tests for Pixiq package."""

import tempfile

from PIL import Image, ImageDraw

from pixiq import Pixiq


def create_test_image(width=1000, height=800):
    """Create a test image for compression."""
    # Create a simple test image
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Draw some shapes to make it more interesting
    draw.rectangle([50, 50, width - 50, height - 50], fill='lightblue', outline='blue', width=5)
    draw.ellipse([200, 200, 400, 400], fill='red', outline='darkred', width=3)
    draw.text((width // 2 - 100, height // 2), 'TEST IMAGE', fill='black')

    return img


def test_basic_compression():
    """Test basic compression functionality."""
    # Create test image
    test_img = create_test_image()

    # Compress image
    result = Pixiq.compress(input=test_img, perceptual_quality=0.9, max_size=800, max_iter=3)

    # Verify compression results
    assert result.selected_quality > 0
    assert result.selected_quality <= 100
    assert result.dimensions[0] <= 800 or result.dimensions[1] <= 800
    assert result.file_size_kb > 0
    assert len(result.iterations_info) > 0


def test_save_thumbnail():
    """Test thumbnail saving functionality."""
    # Create test image
    test_img = create_test_image(800, 600)

    # Compress image
    result = Pixiq.compress(input=test_img, perceptual_quality=0.85, max_size=600)

    # Create thumbnail
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        thumbnail_result = result.save_thumbnail(max_size=200, output=tmp.name)

        # Verify thumbnail results
        assert thumbnail_result.dimensions[0] <= 200 or thumbnail_result.dimensions[1] <= 200
        assert thumbnail_result.selected_quality == result.selected_quality
        assert thumbnail_result.fmt == result.fmt
        assert thumbnail_result.file_size_kb > 0


def test_result_properties():
    """Test CompressionResult properties."""
    test_img = create_test_image(500, 400)
    result = Pixiq.compress(input=test_img, perceptual_quality=0.8, max_iter=2)

    # Test computed properties
    assert isinstance(result.file_size_bytes, int)
    assert isinstance(result.file_size_kb, float)
    assert isinstance(result.dimensions, tuple)
    assert len(result.dimensions) == 2

    # Test iteration info
    assert result.iterations_count >= 0
    if result.iterations_info:
        assert result.last_iteration is not None
        assert result.best_iteration is not None
        assert 'quality' in result.best_iteration
        assert 'error' in result.best_iteration


def test_compress_with_output():
    """Test compression with file output."""
    test_img = create_test_image(300, 200)

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        result = Pixiq.compress(
            input=test_img,
            perceptual_quality=0.85,
            max_size=250,
            output=tmp.name
        )

        # Verify file was created and result is valid
        assert result.selected_quality > 0
        assert result.file_size_kb > 0


def test_compress_validation():
    """Test input validation."""
    import pytest

    # Test invalid perceptual_quality
    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        Pixiq.compress(Image.new('RGB', (100, 100)), perceptual_quality=1.5)

    # Test invalid tolerance
    with pytest.raises(ValueError, match="must be positive"):
        Pixiq.compress(Image.new('RGB', (100, 100)), tolerance=-0.1)

    # Test invalid max_size
    with pytest.raises(ValueError, match="must be positive"):
        Pixiq.compress(Image.new('RGB', (100, 100)), max_size=-100)


def test_save_thumbnail_validation():
    """Test save_thumbnail validation."""
    import pytest

    test_img = create_test_image(200, 200)
    result = Pixiq.compress(input=test_img, perceptual_quality=0.8)

    # Test invalid max_size
    with pytest.raises(ValueError, match="must be positive"):
        result.save_thumbnail(max_size=-50)


def test_format_detection():
    """Test automatic format detection."""
    test_img = create_test_image(200, 200)

    # Test JPEG format detection
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        result = Pixiq.compress(input=test_img, output=tmp.name)
        assert result.fmt == 'JPEG'

    # Test WEBP format detection
    with tempfile.NamedTemporaryFile(suffix='.webp', delete=False) as tmp:
        result = Pixiq.compress(input=test_img, output=tmp.name)
        assert result.fmt == 'WEBP'
