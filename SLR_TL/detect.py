import argparse
import threading
import queue  # 引入队列
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils import read_cfg
from ultralytics import YOLO
import pyttsx3

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=640)
parser.add_argument('--conf', type=float, default=0.5)
parser.add_argument('--device', type=str, default='CUDA GPU', choices=('CUDA GPU', 'openvino'))
args = parser.parse_args()

# 创建 pyttsx3 引擎
engine = pyttsx3.init()
speech_queue = queue.Queue()  # 语音播报队列

def speech_worker():
    """ 独立的线程，负责处理语音队列 """
    while True:
        text = speech_queue.get()  # 阻塞等待新的语音内容
        if text is None:  # 退出信号
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

# 启动语音播报线程
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()

def speak(text):
    """ 将播报内容放入队列 """
    speech_queue.put(text)

def cv2ImgAddText(img, text, position, textColor, textSize=20):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

off_Cfg, on_Cfg, Cfg = read_cfg(os.getcwd())
model_type = off_Cfg['model']
source = off_Cfg['source']
cap = cv2.VideoCapture(source)
print(cap.isOpened())

if args.device == 'openvino':
    model_path = f'models/{model_type}/best_openvino_model'
else:
    model_path = f'models/{model_type}/best.pt'

model = YOLO(model_path, task='detect')

result_list = [('None', 0)]
last_spoken_label = "None"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Done!")
        break

    results = model(frame, conf=args.conf, verbose=False, imgsz=args.size, task='detect')
    lc = []
    result = results[0]
    boxes = result.boxes

    for j in range(boxes.shape[0]):
        label = result.names[int(boxes.cls[j])]
        conf = float(boxes.conf[j])
        lc.append((label, conf))

    if len(lc) > 0:
        max_conf_piece = max(lc, key=lambda x: x[1])
        max_conf_label, max_conf = max_conf_piece
    else:
        max_conf_label, max_conf = 'None', 0

    if max_conf_label != 'None':
        result_list.append((max_conf_label, max_conf))

        # 只在检测到新目标时进行语音播报
        if max_conf_label != last_spoken_label:
            last_spoken_label = max_conf_label
            speak(max_conf_label)  # 直接放入语音队列

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.copyMakeBorder(frame, 80, 80, 0, 0, cv2.BORDER_CONSTANT)
    frame = cv2ImgAddText(frame, f"Detections: {result_list[-1][0]}", (10, 20), (255, 0, 0), 20)
    frame = cv2ImgAddText(frame, f"conf: {result_list[-1][1]:.2f}", (10, 40), (255, 0, 0), 20)

    cv2.imshow("frame", frame)
    cv2.waitKey(1)

    if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
        break

# 结束语音线程
speech_queue.put(None)
speech_thread.join()

print("Done!")
cap.release()
