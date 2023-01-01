import cv2
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from paddleocr import PaddleOCR, draw_ocr


base_dir = os.path.expanduser("~/Videos/chibi-maruko-chinese")
vid_file_names = os.listdir(base_dir)
vid_file_names = sorted(vid_file_names, key=lambda x: int(re.search("#\d+", x).group(0)[1:]))


for vfn in vid_file_names:
    vfp = f"{base_dir}/{vfn}"
    cap = cv2.VideoCapture(vfp)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(n_frames, fps)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret == False:
            break 
        count += 1 
        if count > 5000:
            break 
    print(vfp)
    break

frame_clipped = frame[int(0.75 * h):, ...]
# plt.figure()
# plt.imshow(frame_clipped)
# plt.show()

ocr = PaddleOCR(use_angle_cls=True, lang="ch")
result = ocr.ocr(frame_clipped)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

from PIL import Image
result = result[0]
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(frame_clipped, boxes, txts, scores, font_path="./simfang.ttf")
plt.figure()
plt.imshow(im_show)
plt.show()

# audio text data fusion? 