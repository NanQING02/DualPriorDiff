"""Generate the five masked versions of an image folder.

Usage
-----
    python -m data.make_masks --type center        --input DIR --output DIR
    python -m data.make_masks --type random_path   --input DIR --output DIR
    python -m data.make_masks --type striped       --input DIR --output DIR
    python -m data.make_masks --type sr            --input DIR --output DIR
    python -m data.make_masks --type half          --input DIR --output DIR

Every mode reads all images of ``--input`` and writes the masked image with the
same file name into ``--output``.
"""

import argparse
import os
import random

import numpy as np
from PIL import Image, ImageDraw

IMAGE_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')


def iter_images(folder):
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(IMAGE_EXT):
            yield name, os.path.join(folder, name)


def ensure_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


# --------------------------------------------------------------------------- #
# center mask
# --------------------------------------------------------------------------- #
def mask_center(img, size):
    width, height = img.size
    if size > min(width, height):
        raise ValueError("square size %d is larger than the image (%dx%d)" % (size, width, height))
    left = width // 2 - size // 2
    top = height // 2 - size // 2
    draw = ImageDraw.Draw(img)
    draw.rectangle((left, top, left + size, top + size), fill=(0, 0, 0))
    return img


# --------------------------------------------------------------------------- #
# random path mask
# --------------------------------------------------------------------------- #
def lerp(start, end, t):
    return start + (end - start) * t


def random_route_draw(image, min_brush_size, max_brush_size, route_length, max_interruptions):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    current_x = random.randint(0, width - min_brush_size)
    current_y = random.randint(0, height - min_brush_size)
    current_brush_size = random.randint(min_brush_size, max_brush_size)

    interruptions = 0
    for _ in range(route_length):
        if interruptions < max_interruptions:
            new_x = random.randint(0, width - min_brush_size)
            new_y = random.randint(0, height - min_brush_size)
            new_brush_size = random.randint(min_brush_size, max_brush_size)

            steps = max_brush_size * 2
            for t in range(steps):
                x = lerp(current_x, new_x, t / steps)
                y = lerp(current_y, new_y, t / steps)
                brush_size = lerp(current_brush_size, new_brush_size, t / steps)
                draw.ellipse([x - brush_size // 2, y - brush_size // 2,
                              x + brush_size // 2, y + brush_size // 2], fill='black')

            current_x, current_y = new_x, new_y
            current_brush_size = new_brush_size
            interruptions = 0
        else:
            interruptions += 1

    return image


# --------------------------------------------------------------------------- #
# striped mask
# --------------------------------------------------------------------------- #
def mask_striped(img, stripe_width, stripe_distance):
    arr = np.array(img)
    height, width = arr.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for row in range(0, height, stripe_distance):
        mask[row:row + stripe_width, :] = 255
    arr[mask > 0] = 0
    return Image.fromarray(arr)


# --------------------------------------------------------------------------- #
# 2x super-resolution mask (checkerboard)
# --------------------------------------------------------------------------- #
def mask_sr(img, mask_color=(255, 255, 255)):
    img = img.convert("RGBA")
    width, height = img.size
    pixels = img.load()
    for x in range(width):
        for y in range(height):
            if (x + y) % 2 == 0:
                pixels[x, y] = mask_color
    return img


# --------------------------------------------------------------------------- #
# half mask
# --------------------------------------------------------------------------- #
def mask_half(img, min_ratio, max_ratio):
    width, height = img.size
    draw = ImageDraw.Draw(img)
    direction = random.choice(['top', 'bottom', 'left', 'right'])
    ratio = random.randint(min_ratio, max_ratio)
    if direction == 'top':
        box = (0, 0, width, int(height * ratio / 100))
    elif direction == 'bottom':
        box = (0, height - int(height * ratio / 100), width, height)
    elif direction == 'left':
        box = (0, 0, int(width * ratio / 100), height)
    else:
        box = (width - int(width * ratio / 100), 0, width, height)
    draw.rectangle(box, fill=(0, 0, 0))
    return img


def main():
    parser = argparse.ArgumentParser(description="Generate masked images")
    parser.add_argument("--type", required=True,
                        choices=["center", "random_path", "striped", "sr", "half"])
    parser.add_argument("--input", required=True, help="folder of source images")
    parser.add_argument("--output", required=True, help="folder for the masked images")
    parser.add_argument("--seed", type=int, default=3407)

    # center
    parser.add_argument("--square_size", type=int, default=120,
                        help="center mode: side length in pixels of the masked square")
    # random path
    parser.add_argument("--min_brush_size", type=int, default=10)
    parser.add_argument("--max_brush_size", type=int, default=18)
    parser.add_argument("--route_length", type=int, default=20)
    parser.add_argument("--max_interruptions", type=int, default=10)
    # striped
    parser.add_argument("--stripe_width", type=int, default=4)
    parser.add_argument("--stripe_distance", type=int, default=8)
    # sr
    parser.add_argument("--mask_color", type=str, default="255,255,255",
                        help="sr mode: checkerboard colour, e.g. 255,255,255 or 0,0,0")
    # half
    parser.add_argument("--min_ratio", type=int, default=40, help="half mode: minimum masked percent")
    parser.add_argument("--max_ratio", type=int, default=60, help="half mode: maximum masked percent")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    ensure_dir(args.output)

    if args.type == "sr":
        color = tuple(int(c) for c in args.mask_color.split(","))

    for name, path in iter_images(args.input):
        img = Image.open(path).convert("RGB")

        if args.type == "center":
            out = mask_center(img, args.square_size)
        elif args.type == "random_path":
            out = random_route_draw(img, args.min_brush_size, args.max_brush_size,
                                    args.route_length, args.max_interruptions)
        elif args.type == "striped":
            out = mask_striped(img, args.stripe_width, args.stripe_distance)
        elif args.type == "sr":
            out = mask_sr(img, color)
            if os.path.splitext(name)[1].lower() in (".jpg", ".jpeg"):
                out = out.convert("RGB")
        else:
            out = mask_half(img, args.min_ratio, args.max_ratio)

        out.save(os.path.join(args.output, name))
        print("saved", os.path.join(args.output, name))


if __name__ == "__main__":
    main()
