#!/usr/bin/env python3
"""
🎨 Pixiq Example Script

Demonstrates the key features of the Pixiq image compression library.
Run this script to see Pixiq in action!
"""

from PIL import Image, ImageDraw

from pixiq import Pixiq


def create_demo_image(width=800, height=600):
    """Create a colorful demo image for compression."""
    img = Image.new('RGB', (width, height), color='#1a1a2e')
    draw = ImageDraw.Draw(img)

    # Add some colorful elements
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
    for i, color in enumerate(colors):
        x = (i * 140) % width
        y = (i * 100) % height
        draw.rectangle([x, y, x + 120, y + 80], fill=color, outline='white', width=2)
        draw.text((x + 10, y + 30), f'Color {i + 1}', fill='black')

    # Add text
    draw.text((width // 2 - 150, height // 2), 'PIXIQ DEMO IMAGE', fill='white')
    draw.text((width // 2 - 120, height // 2 + 40), 'Smart Compression', fill='#4ecdc4')

    return img


def main():
    print('🎨 Pixiq - Smart Image Compression Demo')
    print('=' * 50)

    # Create demo image
    print('📷 Creating demo image...')
    original_img = create_demo_image()
    print(f'   Original size: {original_img.size}')

    # Basic compression
    print('\n⚡ Basic compression (90% quality)...')
    result = Pixiq.compress(original_img, perceptual_quality=0.9)
    print(f'   Selected quality: {result.selected_quality}')
    print('.2f')
    print(f'   Compression ratio: {len(original_img.tobytes()) / result.file_size:.1f}x')

    # Compare formats
    print('\n🎨 Comparing formats (85% quality)...')
    formats = ['JPEG', 'WEBP', 'AVIF']
    results = {}

    for fmt in formats:
        try:
            result = Pixiq.compress(original_img, perceptual_quality=0.85, format=fmt)
            results[fmt] = result
            print(f'   {fmt:4}: Quality {result.selected_quality:2}, Size {result.file_size_kb:6.2f} KB')
        except Exception as e:
            print(f'   {fmt:4}: Error - {e}')

    # Thumbnail generation
    print('\n🖼️  Thumbnail generation...')
    if results:
        thumbnail = results['JPEG'].save_thumbnail(max_size=300)
        print(f'   Thumbnail: {thumbnail.dimensions}, Size: {thumbnail.file_size_kb:.2f} KB')

    print('\n🎉 Demo completed! Pixiq makes image compression effortless.')
    print('\n💡 Try it yourself:')
    print('   from pixiq import Pixiq')
    print('   result = Pixiq.compress(your_image, perceptual_quality=0.9)')


if __name__ == '__main__':
    main()
