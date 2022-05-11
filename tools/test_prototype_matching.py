import math

# https://www.geogebra.org/classic
# ggbApplet.getXcoord('H').toFixed() + ", " + -ggbApplet.getYcoord('H').toFixed() + ", " + Math.min(ggbApplet.getValue('l1').toFixed(), ggbApplet.getValue('l2').toFixed()) + ", " + Math.max(ggbApplet.getValue('l1').toFixed(), ggbApplet.getValue('l2').toFixed()) + ", " + (ggbApplet.getValue('α')*180/Math.PI).toFixed()
from pathlib import Path

from PIL import Image

SAMPLES = [
    ('data/ds2_dense/images/lg-2267728-aug-gutenberg1939--page-2.png', [
        {
            'proposal': (1475, 2379, 19, 26, 2 / 180.0 * math.pi),
            'class': 29,  # noteheadHalfOnLine
            'gt': (1477, 2375, 16, 25, 7 / 180.0 * math.pi)
        },
        {
            'proposal': (125, 161, 45, 104, 0 / 180.0 * math.pi),
            'class': 6,  # clefG
            'gt': (127, 164, 44, 101, 5 / 180.0 * math.pi)
        }
    ]),
    ('data/ds2_dense/images/lg-252689430529936624-aug-beethoven--page-3.png', [
        {
            'proposal': (506, 568, 16, 152, 0),
            'class': 123,  # tie
            'gt': (507, 569, 13, 148, 0)
        }
    ])
]

MASKS = {
    6: Image.open('prototype/clefG.png'),
    29: Image.open('prototype/noteheadHalfOnLine.png'),
    123: Image.open('prototype/tie.png'),
}

def process(img, bbox, mask):
    return bbox

def calc_loss(img, bbox, mask) -> float:
    pass

if __name__ == '__main__':
    for image_fp, samples in SAMPLES:
        img = Image.open(image_fp)
        for sample in samples:
            mask = MASKS[sample['class']]
            corr = process(img, sample['proposal'], mask)
            gt_loss = calc_loss(img, sample['gt'], mask)
            new_loss = calc_loss(img, sample['proposal'], mask)

