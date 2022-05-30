import argparse
import math
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image as PImage
from scipy.ndimage.interpolation import rotate

from DeepScoresV2_s2anet.omr_prototype_alignment.render import Render

BASE_PATH = Path("DeepScoresV2_s2anet/omr_prototype_alignment")

class_names = (
    'brace', 'ledgerLine', 'repeatDot', 'segno', 'coda', 'clefG', 'clefCAlto', 'clefCTenor', 'clefF',
    'clefUnpitchedPercussion', 'clef8', 'clef15', 'timeSig0', 'timeSig1', 'timeSig2', 'timeSig3', 'timeSig4',
    'timeSig5', 'timeSig6', 'timeSig7', 'timeSig8', 'timeSig9', 'timeSigCommon', 'timeSigCutCommon',
    'noteheadBlackOnLine', 'noteheadBlackOnLineSmall', 'noteheadBlackInSpace', 'noteheadBlackInSpaceSmall',
    'noteheadHalfOnLine', 'noteheadHalfOnLineSmall', 'noteheadHalfInSpace', 'noteheadHalfInSpaceSmall',
    'noteheadWholeOnLine', 'noteheadWholeOnLineSmall', 'noteheadWholeInSpace', 'noteheadWholeInSpaceSmall',
    'noteheadDoubleWholeOnLine', 'noteheadDoubleWholeOnLineSmall', 'noteheadDoubleWholeInSpace',
    'noteheadDoubleWholeInSpaceSmall', 'augmentationDot', 'stem', 'tremolo1', 'tremolo2', 'tremolo3', 'tremolo4',
    'tremolo5', 'flag8thUp', 'flag8thUpSmall', 'flag16thUp', 'flag32ndUp', 'flag64thUp', 'flag128thUp', 'flag8thDown',
    'flag8thDownSmall', 'flag16thDown', 'flag32ndDown', 'flag64thDown', 'flag128thDown', 'accidentalFlat',
    'accidentalFlatSmall', 'accidentalNatural', 'accidentalNaturalSmall', 'accidentalSharp', 'accidentalSharpSmall',
    'accidentalDoubleSharp', 'accidentalDoubleFlat', 'keyFlat', 'keyNatural', 'keySharp', 'articAccentAbove',
    'articAccentBelow', 'articStaccatoAbove', 'articStaccatoBelow', 'articTenutoAbove', 'articTenutoBelow',
    'articStaccatissimoAbove', 'articStaccatissimoBelow', 'articMarcatoAbove', 'articMarcatoBelow', 'fermataAbove',
    'fermataBelow', 'caesura', 'restDoubleWhole', 'restWhole', 'restHalf', 'restQuarter', 'rest8th', 'rest16th',
    'rest32nd', 'rest64th', 'rest128th', 'restHNr', 'dynamicP', 'dynamicM', 'dynamicF', 'dynamicS', 'dynamicZ',
    'dynamicR', 'graceNoteAcciaccaturaStemUp', 'graceNoteAppoggiaturaStemUp', 'graceNoteAcciaccaturaStemDown',
    'graceNoteAppoggiaturaStemDown', 'ornamentTrill', 'ornamentTurn', 'ornamentTurnInverted', 'ornamentMordent',
    'stringsDownBow', 'stringsUpBow', 'arpeggiato', 'keyboardPedalPed', 'keyboardPedalUp', 'tuplet3', 'tuplet6',
    'fingering0', 'fingering1', 'fingering2', 'fingering3', 'fingering4', 'fingering5', 'slur', 'beam', 'tie',
    'restHBar', 'dynamicCrescendoHairpin', 'dynamicDiminuendoHairpin', 'tuplet1', 'tuplet2', 'tuplet4', 'tuplet5',
    'tuplet7', 'tuplet8', 'tuplet9', 'tupletBracket', 'staff', 'ottavaBracket'
)


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
        self.last_symbol = None
        self.last_symbol_rotated = None
        self.last_glyph_angle = None

    def get_transformed_glyph(self, class_name: str, glyph_width: int, glyph_height: int, glyph_angle: float,
                              padding_left: int, padding_right: int, padding_top: int, padding_bottom: int,
                              svg_path: str = str(BASE_PATH / 'data' / 'Bravura.svg'),
                              csv_path: str = str(BASE_PATH / 'data' / 'name_uni.csv')) -> np.array:
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

        def add_padding(img, top, right, bottom, left):
            new_width = int(right + left)
            new_height = int(top + bottom)
            result = PImage.new(img.mode, (new_width, new_height), (0, 0, 0, 0))
            result.paste(img, (int(left) - img.width // 2, int(top) - img.height // 2))
            return result

        # Assuming the angle is not formatted, if it is comment the next line
        # glyph_angle = glyph_angle / 180.0 * math.pi

        if class_name == self.last_class_name:
            img = self.last_symbol
        else:
            cache_fp = Path(BASE_PATH / 'cache' / f"{class_name}.npy")
            if cache_fp.exists():
                img = np.load(str(cache_fp))
            else:
                case = Render(class_name=class_name, height=glyph_height, width=glyph_width, csv_path=csv_path)
                png_data = case.render(svg_path)
                with BytesIO(png_data) as bio:
                    img = PImage.open(bio)
                    img.load()
                    img = img.transpose(PImage.FLIP_TOP_BOTTOM)
                    img = np.array(img)[..., 3]
                    img = img > 128
                np.save(str(cache_fp), img)

            self.last_class_name = class_name
            self.last_symbol = img.copy()

        if self.last_glyph_angle == glyph_angle:
            img = self.last_symbol_rotated

        else:
            img = rotate(img, angle=glyph_angle * 180.0 / math.pi)
            self.last_symbol_rotated = img.copy()
            self.last_glyph_angle = glyph_angle

        # custom padding - faster than numpy.pad
        img2 = np.zeros((int(padding_top+padding_bottom), int(padding_left + padding_right)), dtype="bool")
        offsets = np.where(img)
        img2[offsets[0] + int(math.floor(padding_top - img.shape[0] / 2)), offsets[1] + int(math.floor(padding_left - img.shape[1] / 2))] = True

        return img2


def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    width, height = 100, 200

    for i in range(100):
        glyph = GlyphGenerator().get_transformed_glyph(class_id=i, glyph_width=width, glyph_height=height,
                                                       glyph_angle=0, padding_top=250, padding_bottom=250,
                                                       padding_right=250, padding_left=250)
        plt.imshow(glyph, cmap="gray")

        ax = plt.gca()
        ax.add_patch(Rectangle((250 - width // 2, 250 - height // 2), width, height, fill=None, alpha=1, color="red",
                               linewidth=3))

        major_ticks = np.arange(0, 500, 50)
        minor_ticks = np.arange(0, 500, 10)

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
