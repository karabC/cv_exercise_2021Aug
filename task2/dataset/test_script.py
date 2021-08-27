import os
import cv2
import argparse
import shutil
import numpy as np
from lxml import etree
from tqdm import tqdm

def dir_create(path):
    if (os.path.exists(path)) and (os.listdir(path) != []):
        shutil.rmtree(path)
        os.makedirs(path)
    if not os.path.exists(path):
        os.makedirs(path)

def parse_args():
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars='@',
        description='Convert CVAT XML annotations to contours'
    )
    parser.add_argument(
            '--image-dir', metavar='DIRECTORY', default='images',
            help='directory with input images'
        )
    parser.add_argument(
            '--cvat-xml', metavar='FILE',default='annotations.xml',
            help='input file with CVAT annotation in xml format'
        )
    parser.add_argument(
            '--output-dir', metavar='DIRECTORY', default='output',
            help='directory for output masks'
        )
    parser.add_argument(
            '--scale-factor', type=float, default=1.0,
            help='choose scale factor for images'
        )
    return parser.parse_args()


def parse_anno_file(cvat_xml, image_name):
    root = etree.parse(cvat_xml).getroot()
    anno = []
    image_id = image_name.split('.')[0]
    image_name_attr = ".//image/[@id='{}']".format(image_id)
    for image_tag in root.iterfind(image_name_attr):
        image = {}
        for key, value in image_tag.items():
            image[key] = value
        image['shapes'] = []
        for box_tag in image_tag.iter('box'):
            box = {'image_id': image_id}
            for key, value in box_tag.items():
                box[key] = value
            if box['label'] == 'head':
                for attribute_tag in box_tag.iter('attribute'): 
                    if attribute_tag.attrib['name'] == 'has_safety_helmet':
                        box['helmet_label'] = attribute_tag.text
                    if attribute_tag.attrib['name'] == 'mask':
                        box['mask_label'] = attribute_tag.text
                image['shapes'].append(box)
        anno.append(image)
    return anno


def create_mask_file(width, height, bitness, background, shapes, scale_factor):
    mask = np.full((height, width, bitness // 8), background, dtype=np.uint8)
    for shape in shapes:
        points = [tuple(map(float, p.split(','))) for p in shape['points'].split(';')]
        points = np.array([(int(p[0]), int(p[1])) for p in points])
        points = points*scale_factor
        points = points.astype(int)
        mask = cv2.drawContours(mask, [points], -1, color=(255, 255, 255), thickness=5)
        mask = cv2.fillPoly(mask, [points], color=(0, 0, 255))
    return mask

def main():
    args = parse_args()
    # args = argparse.ArgumentParser.parse_args(['--image-dir', '..\dateset\images', '--cvat-xml', '..\dateset\annotations.xml', '--output-dir', '..\dataset\output'])
    dir_create(args.output_dir)

    IMAGE_SRC = './dataset/images'
    ANNOTATION_FILE = './dataset/annotations.xml'

    img_list = [f for f in os.listdir(IMAGE_SRC) if os.path.isfile(os.path.join(IMAGE_SRC, f))]
    mask_bitness = 24
    for img in tqdm(img_list, desc='Writing contours:'):
        img_path = os.path.join(IMAGE_SRC, img)
        anno = parse_anno_file(ANNOTATION_FILE, img)
        background = []
        is_first_image = True
        for image in anno:
            if is_first_image:
                current_image = cv2.imread(img_path)
                height, width, _ = current_image.shape
                background = np.zeros((height, width, 3), np.uint8)
                is_first_image = False
            output_path = os.path.join(args.output_dir, img.split('.')[0] + '.png')
            background = create_mask_file(width,
                                          height,
                                          mask_bitness,
                                          background,
                                          image['shapes'],
                                          args.scale_factor)
        cv2.imwrite(output_path, background)


if __name__ == "__main__":
    main()


