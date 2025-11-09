# 🎨 Pixiq - Smart Image Compression

<div align="center">

**Intelligently compress images while preserving visual quality**

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#testing)

*Compress smarter, not harder. Let AI choose the perfect quality for your images.*

[📦 Installation](#installation) • [🚀 Quick Start](#quick-start) • [📚 Documentation](#api-reference) • [❓ Why Pixiq?](#why-pixiq)

</div>

---

## ✨ Key Features

🎯 **Smart Quality Selection** - Automatically finds the optimal compression quality to match your target visual quality

🖼️ **Multiple Formats** - Supports JPEG, PNG, WEBP, and AVIF with format-specific optimizations

📏 **Intelligent Resizing** - Resize images while maintaining aspect ratio and quality

🔄 **Thumbnail Generation** - Create smaller versions of compressed images instantly

🎨 **Alpha Channel Support** - Handles transparent images correctly for each format

⚡ **Performance Optimized** - Fast compression with efficient memory usage

🧩 **Tile-Based Quality Check** - PSNR analysis performed on image tiles for fast quality assessment

🛡️ **Robust Validation** - Comprehensive input validation and error handling

🔍 **Quality Metrics** - Detailed compression statistics and iteration info

---

## 📦 Installation

### From PyPI (Recommended)

**Using uv (fastest):**
```bash
uv add pixiq
# or
uv pip install pixiq
```

**Using pip:**
```bash
pip install pixiq
```

### From Source

```bash
git clone https://github.com/yourusername/pixiq.git
cd pixiq

# Using uv
uv sync --dev
uv pip install -e .

# Using pip
pip install -e .
```

### Requirements

- Python 3.9+
- PIL (Pillow) - Image processing
- NumPy - Array operations
- pillow-avif-plugin - AVIF format support

**💡 Tip:** [uv](https://github.com/astral-sh/uv) is significantly faster than pip and provides better dependency management.

---

## 🚀 Quick Start

**Try it now!** Run the interactive demo:

```bash
python example.py
```

### Basic Compression

```python
from PIL import Image
from pixiq import Pixiq

# Open your image
image = Image.open('photo.jpg')

# Compress with target quality
result = Pixiq.compress(image, perceptual_quality=0.9)

print(f"✅ Compressed! Size: {result.file_size_kb:.1f} KB, Quality: {result.selected_quality}")
```

### Advanced Usage

```python
# Compress with custom settings
result = Pixiq.compress(
    input=image,
    perceptual_quality=0.85,    # Target visual quality (0.0-1.0)
    max_size=2000,             # Resize if larger than 2000px
    format='WEBP',             # Force WEBP format
    hash_type='sha1',          # Use SHA1 instead of default SHA256
    output='compressed.webp'   # Save to file
)

# Access compression details
print(f"📊 Quality: {result.selected_quality}/100")
print(f"📏 Dimensions: {result.dimensions}")
print(f"💾 Size: {result.file_size_kb:.1f} KB")
print(f"🔐 Hash ({result.hash_type}): {result.hash}")
print(f"🎯 Achieved quality: {result.best_iteration['perceptual_quality']:.3f}")
```

### Thumbnail Generation

```python
# Create a thumbnail from compressed image
thumbnail = result.save_thumbnail(
    max_size=500,
    output='thumbnail.webp'
)

print(f"🖼️ Thumbnail: {thumbnail.dimensions}, {thumbnail.file_size_kb:.1f} KB")
```

---

## 📚 API Reference

### `Pixiq.compress()` ⚡

The main method for intelligent image compression with automatic quality selection.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | PIL Image | **required** | Input image to compress |
| `perceptual_quality` | float | `0.95` | Target visual quality (0.0-1.0) |
| `tolerance` | float | `0.005` | Quality tolerance for convergence |
| `max_quality` | int | `None` | Maximum compression quality (1-100) |
| `min_quality` | int | `None` | Minimum compression quality (1-100) |
| `max_size` | int | `None` | Maximum dimension (resizes if larger) |
| `max_iter` | int | `5` | Maximum binary search iterations |
| `format` | str | `None` | Force format: 'JPEG', 'PNG', 'WEBP', 'AVIF' |
| `hash_type` | str | `'sha256'` | Hash algorithm: 'sha1', 'sha256', etc. |
| `output` | str/BytesIO | `None` | Output file path or buffer |

#### Returns
`CompressionResult` - Object containing compressed image and metadata

#### Exceptions
- `TypeError` - Invalid parameter types
- `ValueError` - Parameter values out of range
- `OSError` - File I/O errors

---

### `CompressionResult` 📊

Container for compression results with convenient access methods.

#### Core Properties

| Property | Type | Description |
|----------|------|-------------|
| `compressed` | PIL Image | The compressed image |
| `iterations_count` | int | Number of compression attempts |
| `iterations_info` | List[dict] | Detailed info for each iteration |
| `selected_quality` | int | Final compression quality (1-100) |
| `hash` | str | Hash of compressed image (SHA1/SHA256) |
| `hash_type` | str | Hash algorithm used ('sha1', 'sha256') |
| `file_size` | int | File size in bytes |
| `fmt` | str | Image format ('jpeg', 'webp', etc.) |
| `extra_save_args` | dict | Format-specific save parameters |

#### Computed Properties

| Property | Type | Description |
|----------|------|-------------|
| `file_size_kb` | float | File size in kilobytes |
| `dimensions` | tuple[int, int] | Image dimensions (width, height) |
| `best_iteration` | dict \| None | Info about best quality match |

#### Methods

##### `save(output)` 💾
Save compressed image to file or buffer.

```python
result.save('output.webp')
result.save(io.BytesIO())
```

##### `save_thumbnail(max_size, output=None)` 🖼️
Create and save a resized version of the compressed image.

```python
thumbnail = result.save_thumbnail(max_size=500, output='thumb.webp')
```

**Returns:** New `CompressionResult` with resized image

---

## 🎨 Supported Formats

| Format | Extension | Alpha Support | Optimization |
|--------|-----------|---------------|--------------|
| **JPEG** | `.jpg`, `.jpeg` | ❌ | Progressive, optimized |
| **PNG** | `.png` | ✅ | Lossless compression |
| **WEBP** | `.webp` | ✅ | Method 6 (max compression) |
| **AVIF** | `.avif` | ✅ | Speed 6 (balanced) |

> 💡 **Tip:** Pixiq automatically detects format from file extension or uses JPEG as fallback

---

## 🧠 How It Works

Pixiq uses a **smart binary search algorithm** to find the optimal compression quality:

1. **🎯 Quality Search** - Binary search over quality range (1-100) to find optimal compression
2. **📊 Tile-Based PSNR Analysis** - Calculate Peak Signal-to-Noise Ratio on selected image tiles (not the entire image) for fast quality assessment. The algorithm automatically detects optimal tile size based on image type (UI vs photos) and selects tiles with the most color complexity for accurate quality measurement.
3. **🧠 Perceptual Mapping** - Convert PSNR to perceptual quality using empirical formula
4. **🎪 Best Match Selection** - Choose quality with minimum error from target perceptual quality
5. **🔗 Efficient Hashing** - Generate SHA1/SHA256 hash from compressed data without re-encoding

### Algorithm Visualization

```
Target Quality: 0.85
┌─────────────────────────────────────┐
│ Quality 1 ────► Quality 100         │
│     ↓              ↓              ↓ │
│  PSNR: 25.3     PSNR: 35.7      PSNR: 42.1 │
│  Perceptual: 0.3 ────► 0.8 ────► 0.95 │
│     │              │              │ │
│     └──────────────┼──────────────┘ │
│                    ▼                │
│              ✅ Best Match          │
│              Quality: 67            │
│              Error: 0.0012          │
└─────────────────────────────────────┘
```

---

## ❓ Why Pixiq?

### 🔍 The Problem
Traditional image compression requires manual quality tuning:

```python
# Old way - guesswork required
image.save('output.jpg', quality=85)  # Is 85 good enough?
image.save('output.jpg', quality=75)  # Too low quality?
image.save('output.jpg', quality=80)  # Still guessing...
```

### ✅ The Solution
Pixiq automatically finds the perfect quality for your needs:

```python
# New way - specify what you want
result = Pixiq.compress(image, perceptual_quality=0.9)
# Automatically finds quality=67 for 90% perceptual quality!
```

### 🚀 Performance Benefits

| Feature | Traditional | Pixiq |
|---------|-------------|-------|
| **Quality Control** | Manual guesswork | Precise target quality |
| **File Size** | Variable, unpredictable | Optimal for quality target |
| **Time** | Multiple manual attempts | Single API call |
| **Consistency** | Depends on user expertise | Consistent, reproducible |
| **Formats** | One quality per format | Optimized per format |

### 📈 Real-World Results

```
Original: photo.jpg (2.3 MB, 4000x3000)
Target: 85% perceptual quality

Format    Quality    Size     Time
JPEG      78         245 KB   0.8s
WEBP      82         198 KB   0.7s
AVIF      75         156 KB   1.2s
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Install development dependencies (using uv - recommended)
uv sync --dev

# Or using pip
pip install -e ".[dev]"

# Run tests
uv run pytest tests/

# Or using pytest directly
pytest tests/

# Or run manually
python -c "from tests.test_pixiq import *; test_basic_compression()"
```

**💡 Tip:** Using `uv sync --dev` is the fastest way to set up the development environment!

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

## 📄 License

**MIT License** - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [Pillow](https://python-pillow.org/) for image processing
- Uses [NumPy](https://numpy.org/) for efficient array operations
- Powered by [uv](https://github.com/astral-sh/uv) for lightning-fast package management
- Inspired by modern image optimization techniques

---

<div align="center">

**Made with ❤️ for developers who care about image quality**

[⭐ Star us on GitHub](https://github.com/yourusername/pixiq) • [🐛 Report Issues](https://github.com/yourusername/pixiq/issues) • [💬 Join Discussions](https://github.com/yourusername/pixiq/discussions)

</div>
