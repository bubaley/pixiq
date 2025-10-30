# Pixiq - Intelligent Image Compression

A library for intelligent image compression with perceptual quality preservation.

## Key Features

- Automatic quality selection to achieve target perceptual quality
- Support for JPEG, WEBP, and AVIF formats
- Image resizing capabilities
- Resaving compressed images with new resolutions
- Optimized hash calculation
- Complete input parameter validation
- Convenient methods for working with compression results

## Usage Example

```python
from PIL import Image
from pixiq import Pixiq

# Compress an image
result = Pixiq.compress(
    input=Image.open('input.jpg'),
    perceptual_quality=0.85,  # Target quality (0.0-1.0)
    max_size=2000,            # Maximum dimension size
    max_quality=95,           # Maximum compression quality
    output='output.avif'      # Output file
)

print(f'Selected quality: {result.selected_quality}')
print(f'File size: {result.file_size_kb:.1f} KB')
print(f'Dimensions: {result.dimensions}')

# Get information about the best iteration
best_iter = result.best_iteration
print(f'Best quality: {best_iter["quality"]}, error: {best_iter["error"]:.4f}')

# Resave with different resolution
result_small = result.save_thumbnail(max_size=600, output='output_600.avif')
print(f'600px file size: {result_small.file_size_kb:.1f} KB')
```

## API Reference

### Pixiq.compress()

Main method for compressing images with automatic quality selection.

**Parameters:**
- `input`: PIL Image - input image (required)
- `perceptual_quality`: float = 0.95 - target perceptual quality (0.0-1.0)
- `tolerance`: float = 0.005 - quality tolerance
- `max_quality`: int = None - maximum compression quality (1-100)
- `min_quality`: int = None - minimum compression quality (1-100)
- `max_size`: int = None - maximum image dimension
- `max_iter`: int = 5 - maximum number of search iterations
- `format`: str = None - output file format ('JPEG', 'WEBP', 'AVIF')
- `output`: str or io.BytesIO = None - output file path or buffer

**Returns:** CompressionResult

**Exceptions:**
- `TypeError`: if input parameters have incorrect types
- `ValueError`: if parameter values exceed allowed ranges
- `IOError`: on file saving errors

### CompressionResult

Class representing the result of image compression.

#### Properties:
- `compressed`: PIL Image - compressed image
- `iterations_count`: int - number of compression iterations
- `iterations_info`: list[dict] - information about each iteration
- `selected_quality`: int - selected compression quality
- `hash`: str - MD5 hash of compressed image
- `fmt`: str - file format
- `extra_save_args`: dict - additional save parameters

#### Computed Properties:
- `file_size_bytes`: int - file size in bytes
- `file_size_kb`: float - file size in kilobytes
- `dimensions`: tuple[int, int] - image dimensions (width, height)
- `last_iteration`: dict | None - information about the last iteration
- `best_iteration`: dict | None - information about the best iteration

#### Methods:
- `save_thumbnail(max_size, output=None)` - resaves thumbnail with new size
- `save(output)` - saves image to specified output

### Pixiq.save_thumbnail()

Static method for creating a thumbnail version of a compressed image.

**Parameters:**
- `result`: CompressionResult - result of previous compression
- `max_size`: int - new maximum dimension size
- `output`: str or io.BytesIO = None - output file path or buffer

**Returns:** new CompressionResult

## Supported Formats

- **JPEG**: with optimization and progressive scanning
- **WEBP**: with maximum compression method (method=6)
- **AVIF**: with maximum speed (speed=6)

## Compression Algorithm

1. Binary search over compression quality (1-100)
2. PSNR calculation between original and compressed images
3. PSNR to perceptual quality conversion (empirical formula)
4. Selection of quality with minimum error relative to target quality
5. Optimized hash calculation from compressed buffer

## Performance

- Optimized hash calculation (without re-encoding)
- Efficient memory usage
- Support for large images
- Smart file format detection
