from PIL import Image

from pixiq import CompressionResult, Pixiq

# image = 'tasker.png'
# image = 'gradient.jpg'
image = 'roll.jpg'
ext = 'jpg'


def print_result(result: CompressionResult):
    print(
        f'{result.file_size_kb} KB | tile_size_ratio: {result.tile_size_ratio} | quality: {result.selected_quality} | format: {result.fmt} | iterations: {result.iterations_count}'
    )


result_image = image.split('.')[0] + '.' + ext
result = Pixiq.compress(
    Image.open(f'images/{image}'),
    f'output/{result_image}',
    # max_size=1850,
    perceptual_quality=0.9,
    tolerance=0.0001,
    max_iter=10,
    # tile_size_ratio=0.2,
)
print_result(result)
# kek = result.save_thumbnail(max_size=586, output=f'output/thump_{result_image}')
# print_result(kek)

# 197.13671875 KB | quality: 59 | format: avif | iterations: 5
# 26.025390625 KB | quality: 59 | format: avif | iterations: 0
