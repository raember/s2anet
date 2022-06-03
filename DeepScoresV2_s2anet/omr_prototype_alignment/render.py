import csv
from io import BytesIO
from pathlib import Path
from xml.dom import minidom

import numpy as np
from PIL import Image as PImage
from PIL.ImageOps import flip
from cairosvg import svg2png
from svgpathtools import parse_path

from mmdet.datasets.deepscoresv2 import thresholds

BASE_PATH = Path("DeepScoresV2_s2anet/omr_prototype_alignment")
CACHE_PATH = (BASE_PATH / "cache")

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


class Render:
    def __init__(self, class_name, height, width,
                 csv_path="DeepScoresV2_s2anet/omr_prototype_alignment/data/name_uni.csv"):
        super(Render, self).__init__()
        self.csv_path = csv_path
        self.class_name = class_name
        self.height = height
        self.width = width
        self.name_uni = self.csv2dict(1, 2)
        self.id_to_class = self.csv2dict(0, 1)

    def csv2dict(self, key_pos, val_pos):
        reader = csv.reader(open(self.csv_path, 'r'))
        d = {}
        for row in reader:
            d[row[key_pos]] = row[val_pos]
        return d

    def create_svg(self, bbox, base_path):
        ht = str(abs(bbox[3] - bbox[2]))
        wt = str(abs(bbox[1] - bbox[0]))

        try:
            index = np.argwhere(np.array(bbox) < 0)[0][0]
        except IndexError:
            index = -1

        tfr = ""
        if index == -1:
            tfr = "translate(0 0)"
        elif index < 2:
            tfr = "translate({0} 0)".format(abs(bbox[index]))
        else:
            tfr = "translate(0 {0})".format(abs(bbox[index]))

        root = minidom.Document()

        xml = root.createElement('svg')
        xml.setAttribute('width', wt)
        xml.setAttribute('height', ht)
        xml.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
        xml.setAttribute('xlink', 'http://www.w3.org/1999/xlink')
        root.appendChild(xml)

        productChild = root.createElement('path')
        productChild.setAttribute('fill', '#000000')
        productChild.setAttribute('fill-rule', 'nonzero')
        productChild.setAttribute('transform', tfr)
        productChild.setAttribute('d', base_path)

        xml.appendChild(productChild)
        xml_str = root.toprettyxml(indent="\t")
        return xml_str

    def render(self, bravura_path, save_svg=True, save_png=True):
        file = minidom.parse(bravura_path)

        glyphs = file.getElementsByTagName('glyph')
        glyph_names = [glyph.attributes["glyph-name"].value for glyph in glyphs]
        name_copy = self.class_name
        if name_copy == 'tupletBracket':
            name_copy = 'beam'
        elif name_copy == 'slur':
            name_copy = 'tie'
        index_uni = glyph_names.index(self.name_uni[name_copy])
        base_path = glyphs[index_uni].attributes['d'].value
        if name_copy == 'tie':
            # example for 50x10
            # m 0 0 q 25 10 50 10 t 50 -10 q -25 8 -50 8 T 0 0 z
            base_path = f"m 0 0 " \
                        f"q {self.width / 4.0} {self.height} {self.width / 2.0} {self.height} " \
                        f"t {self.width / 2.0} -{self.height} " \
                        f"q -{self.width / 4.0} {self.height - 2} -{self.width / 2.0} {self.height - 2} " \
                        f"T 0 0 z"
            bbox = [0, self.width, 0, self.height]
        # elif name_copy == 'slur':
        #     base_path = "m 154,141.7 1.2,1.3 C 140,155 114.7,158 95.8,158 64.5,158 49.6,150.8 40,143.3 l 1.4,-1.9 c 6.7,7.3 33.4,11.7 55.3,11.7 25.2,0 41.9,-2.8 57.3,-11.3 z"
        #     path_alt = parse_path(base_path)
        #     bbox = path_alt.bbox()
        else:
            path_alt = parse_path(base_path)
            bbox = path_alt.bbox()

        xml_str = self.create_svg(bbox, base_path)

        png_data = None
        if save_png:
            out_folder = Path('png_files')
            out_folder.mkdir(exist_ok=True)
            png_path = out_folder / self.class_name
            # svg2png(bytestring=xml_str.encode(), write_to=str(png_path.with_suffix('.png')), output_width=self.width,
            #         output_height=self.height)
            png_data = svg2png(bytestring=xml_str.encode(), output_width=self.width, output_height=self.height)
            with BytesIO(png_data) as bio:
                img = PImage.open(bio)
                img.load()
                # img = img.rotate(a * 180.0 / math.pi, PImage.BILINEAR, expand=True, fillcolor=(0, 0, 0, 0))
                img = img.transpose(PImage.FLIP_TOP_BOTTOM)
                img.save(str(png_path.with_suffix('.png')))
        else:
            png_data = svg2png(bytestring=xml_str.encode(), output_width=self.width, output_height=self.height)

        if save_svg:
            out_folder = Path('svg_files')
            out_folder.mkdir(exist_ok=True)
            save_path_file = out_folder / self.class_name
            with open(save_path_file.with_suffix('.svg'), "w") as f:
                f.write(xml_str)

        return png_data


def remove_padding(img):
    x_values = np.where(np.max(img, axis=0) > 0)
    y_values = np.where(np.max(img, axis=1) > 0)

    return img[np.min(y_values):np.max(y_values) + 1, np.min(x_values):np.max(x_values) + 1]


def fill_cache():
    needs_flip = ['clefF']

    if not CACHE_PATH.exists():
        CACHE_PATH.mkdir()

        for name, params in zip(class_names, thresholds.values()):
            if len(params) == 0:
                continue
            l1 = list(params[2]) if isinstance(params[2], tuple) else [params[2]]
            l2 = list(params[3]) if isinstance(params[2], tuple) else [params[3]]
            size = max(l1 + l2)
            png_data = Render(
                class_name=name, height=size, width=size,
                csv_path=str(BASE_PATH / 'data' / 'name_uni.csv')).render(
                str(BASE_PATH / 'data' / 'Bravura.svg'), save_svg=False, save_png=False)

            with BytesIO(png_data) as bio:
                img = PImage.open(bio)
                img.load()
                img = img.transpose(PImage.FLIP_TOP_BOTTOM)
                if name in needs_flip:
                    img = flip(img)

                img = np.array(img)[..., 3]
                img = remove_padding(img)

                np.save(CACHE_PATH / name, img)


if __name__ == '__main__':
    fill_cache()
