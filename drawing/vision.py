"""Shared image preparation for model prompts and training examples."""
from drawing.config import IMAGE_SIZE


def prepare_image(image, processor, image_size=IMAGE_SIZE, vision='qwen'):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if vision == 'kimi':
        image.thumbnail((int(image_size['longest_edge'] ** 0.5),) * 2)
        return image
    if vision != 'qwen':
        raise ValueError(f'Unsupported vision profile: {vision}')
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    height, width = smart_resize(image.height, image.width,
        factor=processor.patch_size * processor.merge_size,
        min_pixels=image_size['shortest_edge'], max_pixels=image_size['longest_edge'])
    return image if image.size == (width, height) else image.resize((width, height))
