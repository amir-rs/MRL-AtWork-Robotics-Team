#!/usr/bin/env python3
# coding: utf-8
"""
Usage:

python3 evaluate_detection.py \
  --predicted_coco /path/to/predicted_detections.json \
  --groundtruth_coco /robotto_objects/test/coco_annotations.json
"""

import argparse

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pycocotools import coco
coco.unicode = bytes  # https://github.com/cocodataset/cocoapi/issues/49

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_arguments():
  """ Setup and parse commandline arguments """
  parser = argparse.ArgumentParser(description="Evaluate detection performance")
  parser.add_argument(
      "--predicted_coco",
      type=Path,
      default=Path("detection_results.json"),
      help="Coco annotations json containing predictions")
  parser.add_argument(
      "--groundtruth_coco",
      type=Path,
      default=Path("/robotto_objects/test/coco_annotations.json"),
      help="Coco annotations json containing groundtruth")

  args = parser.parse_args()

  return args


def main():
  annType = 'bbox'

  cocoGt = COCO(str(ARGS.groundtruth_coco))
  cocoDt = cocoGt.loadRes(str(ARGS.predicted_coco))

  cocoEval = COCOeval(cocoGt, cocoDt, annType)
  cocoEval.evaluate()
  cocoEval.accumulate()
  cocoEval.summarize()


if __name__ == "__main__":
  ARGS = parse_arguments()
  main()
