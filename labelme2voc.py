import json
import os
import sys
import glob
import argparse

import imgviz
import labelme
import numpy as np

from PIL import Image, ImageDraw


def shape_to_mask(img_shape, points):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]

        cls_id = label_name_to_value[label]
        mask = shape_to_mask(img_shape[:2], points)

        cls[mask] = cls_id

    return cls


def main(args):
    if os.path.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)

    os.makedirs(args.output_dir)
    os.makedirs(os.path.join(args.output_dir, "JPEGImages"))
    os.makedirs(os.path.join(args.output_dir, "SegmentationClassPNG"))

    if not args.no_viz:
        os.makedirs(os.path.join(args.output_dir, "SegmentationClassVisualization"))

    print("Creating dataset:", args.output_dir)

    class_names = []
    class_name_to_id = {}
    with open(args.labels, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            class_id = i - 1  # starts with -1
            class_name = line.strip()
            class_name_to_id[class_name] = class_id
            if class_id == -1:
                assert class_name == "__ignore__"
                continue
            elif class_id == 0:
                assert class_name == "__background__"
            class_names.append(class_name)
    print("class_names:", class_names)

    out_class_names_file = os.path.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    for filename in os.listdir(args.input_dir):
        print("Generating dataset from:", filename)

        with open(filename, "r") as f:
            json_file = json.load(f)
            shapes = [dict(label=x["label"], points=x["points"]) for x in json_file["shapes"]]

        image_filename = filename.replace(".json", ".bmp")
        image_pil = Image.open(image_filename)

        base = os.path.splitext(os.path.basename(filename))[0]
        out_img_file = os.path.join(args.output_dir, "JPEGImages", base + ".jpg")
        out_png_file = os.path.join(args.output_dir, "SegmentationClassPNG", base + ".png")

        if not args.no_viz:
            out_viz_file = os.path.join(args.output_dir, "SegmentationClassVisualization", base + ".jpg")

        image_pil.save(out_img_file)
        image_arr = np.array(image_pil)

        lbl = shapes_to_label(img_shape=image_arr.shape, shapes=shapes, label_name_to_value=class_name_to_id)
        labelme.utils.lblsave(out_png_file, lbl)

        if not args.no_viz:
            viz = imgviz.label2rgb(
                lbl,
                imgviz.rgb2gray(image_arr),
                font_size=15,
                label_names=class_names,
                loc="rb",
            )
            imgviz.io.imsave(out_viz_file, viz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("LabelMe annotation converter to image train")

    parser.add_argument("--labels", type=str, default="./wheel_data/data/labels.txt", help="labels file")
    parser.add_argument("--input-dir", type=str, default="./wheel_data/data", help="input annotated directory")
    parser.add_argument("--output-dir", type=str, default="./results", help="output dataset directory")
    parser.add_argument("--no-viz", help="no visualization", action="store_true")

    opt = parser.parse_args()
    main(opt)
