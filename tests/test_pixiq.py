"""Tests for Pixiq package."""

import tempfile

from PIL import Image, ImageDraw

from pixiq import Pixiq


def create_test_image(width=1000, height=800, mode='RGB'):
    """Create a test image for compression."""
    # Ensure minimum size for drawing operations
    draw_width = max(10, width // 10)
    draw_height = max(10, height // 10)

    if mode == 'RGBA':
        # Create RGBA image with alpha channel
        img = Image.new('RGBA', (width, height), color=(255, 255, 255, 128))
        draw = ImageDraw.Draw(img)

        # Draw some shapes with alpha
        draw.rectangle(
            [draw_width, draw_height, width - draw_width, height - draw_height],
            fill=(173, 216, 230, 200),
            outline=(0, 0, 255, 255),
            width=2,
        )
        if width > 50 and height > 50:
            draw.ellipse(
                [width // 4, height // 4, width // 2, height // 2],
                fill=(255, 0, 0, 180),
                outline=(139, 0, 0, 255),
                width=2,
            )
        if width > 20:
            draw.text((max(0, width // 2 - 50), max(0, height // 2)), 'TEST', fill=(0, 0, 0, 255))
    elif mode == 'P':
        # Create palette image (simulating GIF/PNG with palette)
        img = Image.new('P', (width, height))
        palette = []
        # Create a simple palette
        for i in range(256):
            palette.extend([i, i, i])  # Grayscale palette
        img.putpalette(palette)

        # Add transparency to palette mode
        img.info['transparency'] = 0  # Make index 0 transparent

        draw = ImageDraw.Draw(img)
        draw.rectangle(
            [draw_width, draw_height, width - draw_width, height - draw_height], fill=100, outline=200, width=2
        )
        if width > 50 and height > 50:
            draw.ellipse([width // 4, height // 4, width // 2, height // 2], fill=150, outline=250, width=2)
        if width > 20:
            draw.text((max(0, width // 2 - 50), max(0, height // 2)), 'TEST', fill=255)
    else:
        # Create RGB image
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        # Draw some shapes to make it more interesting
        draw.rectangle(
            [draw_width, draw_height, width - draw_width, height - draw_height],
            fill='lightblue',
            outline='blue',
            width=2,
        )
        if width > 50 and height > 50:
            draw.ellipse([width // 4, height // 4, width // 2, height // 2], fill='red', outline='darkred', width=2)
        if width > 20:
            draw.text((max(0, width // 2 - 50), max(0, height // 2)), 'TEST', fill='black')

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
    assert isinstance(result.file_size, int)
    assert isinstance(result.file_size_kb, float)
    assert isinstance(result.dimensions, tuple)
    assert len(result.dimensions) == 2

    # Test iteration info
    assert result.iterations_count >= 0
    if result.iterations_info:
        assert result.best_iteration is not None
        assert 'quality' in result.best_iteration
        assert 'error' in result.best_iteration


def test_compress_with_output():
    """Test compression with file output."""
    test_img = create_test_image(300, 200)

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        result = Pixiq.compress(input=test_img, perceptual_quality=0.85, max_size=250, output=tmp.name)

        # Verify file was created and result is valid
        assert result.selected_quality > 0
        assert result.file_size_kb > 0


def test_compress_validation():
    """Test input validation."""
    # Test invalid perceptual_quality
    try:
        Pixiq.compress(Image.new('RGB', (100, 100)), perceptual_quality=1.5)
        assert False, 'Should have raised ValueError for perceptual_quality > 1.0'
    except ValueError as e:
        assert 'between 0.0 and 1.0' in str(e)

    # Test invalid tolerance
    try:
        Pixiq.compress(Image.new('RGB', (100, 100)), tolerance=-0.1)
        assert False, 'Should have raised ValueError for negative tolerance'
    except ValueError as e:
        assert 'must be positive' in str(e)

    # Test invalid max_size
    try:
        Pixiq.compress(Image.new('RGB', (100, 100)), max_size=-100)
        assert False, 'Should have raised ValueError for negative max_size'
    except ValueError as e:
        assert 'must be positive' in str(e)


def test_save_thumbnail_validation():
    """Test save_thumbnail validation."""
    test_img = create_test_image(200, 200)
    result = Pixiq.compress(input=test_img, perceptual_quality=0.8)

    # Test invalid max_size
    try:
        result.save_thumbnail(max_size=-50)
        assert False, 'Should have raised ValueError for negative max_size'
    except ValueError as e:
        assert 'must be positive' in str(e)


def test_format_detection():
    """Test automatic format detection."""
    test_img = create_test_image(200, 200)

    # Test JPEG format detection
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        result = Pixiq.compress(input=test_img, output=tmp.name)
        assert result.fmt == 'jpeg'

    # Test WEBP format detection
    with tempfile.NamedTemporaryFile(suffix='.webp', delete=False) as tmp:
        result = Pixiq.compress(input=test_img, output=tmp.name)
        assert result.fmt == 'webp'

    # Test PNG format detection
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        result = Pixiq.compress(input=test_img, output=tmp.name)
        assert result.fmt == 'png'

    # Test AVIF format detection
    with tempfile.NamedTemporaryFile(suffix='.avif', delete=False) as tmp:
        result = Pixiq.compress(input=test_img, output=tmp.name)
        assert result.fmt == 'avif'


def test_explicit_format_parameter():
    """Test explicit format parameter overrides file extension."""
    test_img = create_test_image(200, 200)

    # Force JPEG format even with .png extension
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        result = Pixiq.compress(input=test_img, output=tmp.name, format='JPEG')
        assert result.fmt == 'jpeg'

    # Force PNG format even with .jpg extension
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        result = Pixiq.compress(input=test_img, output=tmp.name, format='PNG')
        assert result.fmt == 'png'


def test_alpha_channel_handling():
    """Test compression of images with alpha channels."""
    # Test RGBA image
    rgba_img = create_test_image(300, 200, mode='RGBA')

    # Test with PNG (supports alpha)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        result = Pixiq.compress(input=rgba_img, output=tmp.name, perceptual_quality=0.8)
        assert result.fmt == 'png'
        assert result.selected_quality > 0
        assert result.file_size > 0

    # Test with WEBP (supports alpha)
    with tempfile.NamedTemporaryFile(suffix='.webp', delete=False) as tmp:
        result = Pixiq.compress(input=rgba_img, output=tmp.name, perceptual_quality=0.8)
        assert result.fmt == 'webp'
        assert result.selected_quality > 0

    # Test with AVIF (supports alpha)
    with tempfile.NamedTemporaryFile(suffix='.avif', delete=False) as tmp:
        result = Pixiq.compress(input=rgba_img, output=tmp.name, perceptual_quality=0.8)
        assert result.fmt == 'avif'
        assert result.selected_quality > 0

    # Test with JPEG (doesn't support alpha) - should convert to RGB
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        result = Pixiq.compress(input=rgba_img, output=tmp.name, perceptual_quality=0.8)
        assert result.fmt == 'jpeg'
        assert result.selected_quality > 0


def test_palette_image_with_transparency():
    """Test compression of palette images with transparency."""
    palette_img = create_test_image(300, 200, mode='P')

    # Test with PNG (supports palette with transparency)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        result = Pixiq.compress(input=palette_img, output=tmp.name, perceptual_quality=0.8)
        assert result.fmt == 'png'
        assert result.selected_quality > 0
        assert result.file_size > 0


def test_bytesio_output():
    """Test compression with BytesIO output."""
    import io

    test_img = create_test_image(300, 200)

    # Test compression to BytesIO
    output_buffer = io.BytesIO()
    result = Pixiq.compress(input=test_img, perceptual_quality=0.8, output=output_buffer)

    # Verify result
    assert result.selected_quality > 0
    assert result.file_size > 0
    assert result.fmt == 'jpeg'  # Default format when no extension

    # Verify buffer contains data (after compression, buffer is at position 0)
    buffer_data = output_buffer.getvalue()
    assert len(buffer_data) > 0
    assert len(buffer_data) == result.file_size

    # Verify we can read the compressed image back
    output_buffer.seek(0)
    compressed_img = Image.open(output_buffer)
    assert compressed_img.size[0] <= 300
    assert compressed_img.size[1] <= 200


def test_save_method():
    """Test CompressionResult.save() method."""
    import io

    test_img = create_test_image(300, 200)
    result = Pixiq.compress(input=test_img, perceptual_quality=0.8)

    # Test saving to file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        result.save(tmp.name)

        # Verify file was created and can be opened
        saved_img = Image.open(tmp.name)
        assert saved_img.size[0] <= 300
        assert saved_img.size[1] <= 200

    # Test saving to BytesIO
    output_buffer = io.BytesIO()
    result.save(output_buffer)

    # Verify buffer contains data (save() leaves buffer at position 0)
    buffer_data = output_buffer.getvalue()
    assert len(buffer_data) > 0

    # Verify we can read the saved image back
    output_buffer.seek(0)
    saved_img = Image.open(output_buffer)
    assert saved_img.size[0] <= 300
    assert saved_img.size[1] <= 200


def test_quality_bounds():
    """Test min_quality and max_quality parameters."""
    test_img = create_test_image(300, 200)

    # Test with min_quality and max_quality
    result = Pixiq.compress(input=test_img, perceptual_quality=0.8, min_quality=20, max_quality=80, max_iter=3)
    assert 20 <= result.selected_quality <= 80

    # Test with only min_quality
    result = Pixiq.compress(input=test_img, perceptual_quality=0.8, min_quality=30, max_iter=2)
    assert result.selected_quality >= 30

    # Test with only max_quality
    result = Pixiq.compress(input=test_img, perceptual_quality=0.8, max_quality=70, max_iter=2)
    assert result.selected_quality <= 70

    # Test validation: max_quality < min_quality
    try:
        Pixiq.compress(input=test_img, min_quality=80, max_quality=20)
        assert False, 'Should have raised ValueError for max_quality < min_quality'
    except ValueError as e:
        assert 'Max quality must be greater than or equal to min quality' in str(e)


def test_tolerance_values():
    """Test different tolerance values affect convergence."""
    test_img = create_test_image(300, 200)

    # Test with loose tolerance (should converge quickly)
    result = Pixiq.compress(input=test_img, perceptual_quality=0.8, tolerance=0.1, max_iter=10)
    assert result.iterations_count >= 1

    # Test with tight tolerance (may take more iterations)
    result = Pixiq.compress(input=test_img, perceptual_quality=0.8, tolerance=0.001, max_iter=10)
    assert result.iterations_count >= 1

    # Test with very tight tolerance
    result = Pixiq.compress(input=test_img, perceptual_quality=0.8, tolerance=1e-6, max_iter=5)
    assert result.iterations_count >= 1


def test_get_image_hash():
    """Test get_image_hash function."""
    # Test that identical images have same hash
    img1 = create_test_image(100, 100)
    img2 = create_test_image(100, 100)

    hash1 = Pixiq.get_image_hash(img1)
    hash2 = Pixiq.get_image_hash(img2)

    # Images should be identical and have same hash
    assert hash1 == hash2
    assert isinstance(hash1, str)
    assert len(hash1) == 64  # SHA256 hex length

    # Test that different images have different hashes
    img3 = Image.new('RGB', (100, 100), color='red')
    hash3 = Pixiq.get_image_hash(img3)
    assert hash1 != hash3

    # Test with different formats
    rgba_img = create_test_image(100, 100, mode='RGBA')
    rgba_hash = Pixiq.get_image_hash(rgba_img)
    assert isinstance(rgba_hash, str)
    assert len(rgba_hash) == 64


def test_hash_functionality_in_compression():
    """Test hash functionality in compression and thumbnail generation."""
    test_img = create_test_image(300, 200)

    # Test with SHA1 hash
    result_sha1 = Pixiq.compress(input=test_img, perceptual_quality=0.85, max_size=250, hash_type='sha1')

    # Verify SHA1 hash properties
    assert result_sha1.hash_type == 'sha1'
    assert len(result_sha1.hash) == 40  # SHA1 hex length
    assert all(c in '0123456789abcdef' for c in result_sha1.hash)
    assert isinstance(result_sha1.hash, str)

    # Test thumbnail generation maintains hash type
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        thumb_result = result_sha1.save_thumbnail(max_size=100, output=tmp.name)

        assert thumb_result.hash_type == 'sha1'
        assert len(thumb_result.hash) == 40
        assert all(c in '0123456789abcdef' for c in thumb_result.hash)
        assert isinstance(thumb_result.hash, str)

        # Different images should have different hashes
        assert result_sha1.hash != thumb_result.hash

    # Test with SHA256 hash
    result_sha256 = Pixiq.compress(input=test_img, perceptual_quality=0.85, max_size=250, hash_type='sha256')

    # Verify SHA256 hash properties
    assert result_sha256.hash_type == 'sha256'
    assert len(result_sha256.hash) == 64  # SHA256 hex length
    assert all(c in '0123456789abcdef' for c in result_sha256.hash)
    assert isinstance(result_sha256.hash, str)

    # Test thumbnail generation maintains hash type
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        thumb_result_sha256 = result_sha256.save_thumbnail(max_size=100, output=tmp.name)

        assert thumb_result_sha256.hash_type == 'sha256'
        assert len(thumb_result_sha256.hash) == 64
        assert all(c in '0123456789abcdef' for c in thumb_result_sha256.hash)
        assert isinstance(thumb_result_sha256.hash, str)

        # Different images should have different hashes
        assert result_sha256.hash != thumb_result_sha256.hash

    # SHA1 and SHA256 hashes should be different even for same image
    assert result_sha1.hash != result_sha256.hash
