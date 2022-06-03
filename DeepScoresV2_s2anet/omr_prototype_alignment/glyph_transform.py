import argparse
import math

import cv2
import numpy as np
from scipy.ndimage.interpolation import rotate

from DeepScoresV2_s2anet.omr_prototype_alignment.render import BASE_PATH, CACHE_PATH


def parse_args():
    parser = argparse.ArgumentParser(description='Glyph transformation for effective post processing')
    parser.add_argument('--class_id', help='Id of the DeepScore Class', type=int, default=5)
    parser.add_argument('--csv_path', help='the path where name_uni.csv is stored', type=str,
                        default=str(BASE_PATH / 'data' / 'name_uni.csv'))
    parser.add_argument('--glyph_height', help='height of glyph', type=int, default=254)
    parser.add_argument('--glyph_width', help='width of glyph', type=int, default=236)
    parser.add_argument('--glyph_angle', help='angle of rotation of glyph', type=float, default=0.0)
    parser.add_argument('--svg_path', help='the path where Bravura.svg is stored', type=str,
                        default=str(BASE_PATH / 'data' / 'Bravura.svg'))
    parser.add_argument('--padding_left', help='Length of padding on the left of glyph', type=int, default=254)
    parser.add_argument('--padding_right', help='Length of padding on the right of glyph', type=int, default=254)
    parser.add_argument('--padding_top', help='Length of padding on the top of glyph', type=int, default=254)
    parser.add_argument('--padding_bottom', help='Length of padding on the bottom of glyph', type=int, default=254)

    args = parser.parse_args()

    return args


class GlyphGenerator:

    def __init__(self):
        self.last_class_name = None
        self.last_width = None
        self.last_height = None
        self.last_glyph_angle = None
        self.symbols = {}
        self.last_symbol_resized = None
        self.last_symbol_resized_rotated = None


    def get_transformed_glyph(self, class_name: str, glyph_width: int, glyph_height: int, glyph_angle: float,
                              padding_left: int, padding_right: int, padding_top: int, padding_bottom: int) -> np.array:
        """
        returns a glyph according the parameters

        :param class_name: The class (type of the glyph)
        :param glyph_width: width of the glyph
        :param glyph_height: height of the glyph
        :param glyph_angle: angle of the glyph
        :param padding_left: padding along the horizontal axis, padding on the left side of the glyph center
        :param padding_right: padding along the horizontal axis, padding on the right side of the glyph center
        :param padding_top: padding along vertical axis, padding above the glyph
        :param padding_bottom: padding along vertical axis, padding below the glyph
        :return: numpy array with the glyph
        """

        # Assuming the angle is not formatted, if it is comment the next line
        # glyph_angle = glyph_angle / 180.0 * math.pi

        do_rotate, do_resize = False, False

        if class_name == self.last_class_name and glyph_width == self.last_width and glyph_height == self.last_height and self.last_glyph_angle == glyph_angle:
            img = self.last_symbol_resized_rotated

        elif class_name == self.last_class_name and glyph_width == self.last_width and glyph_height == self.last_height:
            img = self.last_symbol_resized
            do_rotate = True
        elif class_name == self.last_class_name:
            img = self.last_symbol
            do_rotate, do_resize = True, True
        else:
            if class_name not in self.symbols:
                self.symbols[class_name] = np.load(str(CACHE_PATH / class_name) + ".npy")
            img = self.symbols[class_name].copy()
            self.last_class_name = class_name
            do_rotate, do_resize = True, True

        if do_resize:
            img = cv2.resize(img, dsize=(glyph_width, glyph_height), interpolation=cv2.INTER_LINEAR)
            self.last_symbol_resized = img.copy()
            self.last_width = glyph_width
            self.last_height = glyph_height

        if do_rotate:
            img = rotate(img, angle=glyph_angle * 180.0 / math.pi)
            self.last_symbol_resized_rotated = img.copy()
            self.last_glyph_angle = glyph_angle

        if padding_left is not None and padding_right is not None and padding_top is not None and padding_bottom is not None:
            # custom padding - faster than numpy.pad
            img2 = np.zeros((int(padding_top + padding_bottom), int(padding_left + padding_right)), dtype="bool")
            offsets = np.where(img)
            img2[offsets[0] + int(math.floor(padding_top - img.shape[0] / 2)), offsets[1] + int(
                math.floor(padding_left - img.shape[1] / 2))] = True
        else:
            img2 = img.copy()

        return img2


def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    width, height = 26, 54
    padding = None

    for class_ in ['clefG']:
        glyph = GlyphGenerator().get_transformed_glyph(class_, glyph_width=width, glyph_height=height,
                                                       glyph_angle=-0.03, padding_top=padding, padding_bottom=padding,
                                                       padding_right=padding, padding_left=padding)
        plt.imshow(glyph, cmap="gray")

        if padding is not None:
            ax = plt.gca()
            ax.add_patch(
                Rectangle((padding - width // 2, padding - height // 2), width, height, fill=None, alpha=1, color="red",
                          linewidth=3))

            major_ticks = np.arange(0, 2 * padding, 50)
            minor_ticks = np.arange(0, 2 * padding, 10)

            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_yticks(major_ticks)
            ax.set_yticks(minor_ticks, minor=True)

        plt.title(f"Width: {width}, Height: {height}")

        plt.tight_layout()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    main()
