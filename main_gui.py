#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from pathlib import Path
from argparse import ArgumentParser
import train_net
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk


def slider_scroll(event):
    scale_var.set(float(event))
    update(filename)


def show_image(event):
    global filename
    filename = str(event.data)
    print(filename)

    update(filename)


def update(filename):
    global display_image, scale_var
    if filename is None:
        return

    # image loading
    rgb_img = cv2.imread(filename)
    scale = scale_var.get() / 100.0
    rgb_img = cv2.resize(rgb_img, None, fx=scale, fy=scale)

    # segmentation
    predictions = model(rgb_img)
    panoptic_seg, segments_info = predictions["panoptic_seg"]

    # visualization
    visualizer = Visualizer(
        rgb_img[:, :, ::-1],
        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
        scale=1.2
    )

    vis_output = visualizer.draw_panoptic_seg_predictions(
        panoptic_seg.to("cpu"), segments_info
    )

    # save
    fname = args.output / Path(filename).name
    vis_output.save(fname)

    # visualize
    vis_image = Image.open(open(fname, "rb"))
    vis_image.thumbnail((1000, 1000), Image.ANTIALIAS)
    display_image = ImageTk.PhotoImage(vis_image)

    canvas.delete("image")
    canvas.create_image(0, 0, image=display_image, tag="segm", anchor=tk.NW)
    canvas.config(width=vis_image.width, height=vis_image.height)
    canvas.pack()
    root.geometry("")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("output", type=Path, help="output directory")
    parser.add_argument("--scale", type=float, help="image scale", default=1.0)
    args = parser.parse_args()

    # parameters for Detectron2
    args_det = train_net.default_argument_parser().parse_args()
    args_det.config_file = "panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml"
    args_det.opts = ["MODEL.WEIGHTS", "model_final_5e6da2.pkl"]
    cfg = train_net.setup(args_det)
    model = DefaultPredictor(cfg)

    args.output.mkdir(parents=True, exist_ok=True)
    filename = None

    # Build main window
    display_image = None

    root = TkinterDnD.Tk()
    root.title('Segmentation')
    root.geometry('1000x1000')
    root.config(bg='#888888')
    root.drop_target_register(DND_FILES)
    root.dnd_bind('<<Drop>>', show_image)

    canvas = tk.Canvas(
        root,
        width=1000,
        height=1000
    )
    canvas.pack()

    root2 = TkinterDnD.Tk()
    scale_var = tk.DoubleVar()
    scale_var.set(args.scale * 100)
    scaleH = tk.Scale(root2,
                      variable=scale_var,
                      command=slider_scroll,
                      orient=tk.HORIZONTAL,  # direction
                      length=300,  # length
                      width=20,  # width
                      sliderlength=20,  # width of the slider
                      from_=0.1,  # min value
                      to=100,  # max value
                      resolution=0.5
                      )
    scaleH.pack()

    root.mainloop()
