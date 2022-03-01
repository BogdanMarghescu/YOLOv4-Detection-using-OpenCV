import os
import cv2
from pathlib import Path
from numpy import loadtxt

class_list = [cls.strip() for cls in open("yolov4_utils/coco_classes.txt")]  # COCO classes
color_list = loadtxt("yolov4_utils/colors.txt").tolist()  # box colors
weightsPath = "yolov4_utils/yolov4.weights"
configPath = "yolov4_utils/yolov4.cfg"
images_path = "images"
videos_path = "videos"
output_images_path = "output_images"
output_videos_path = "output_video"
images_dir = Path(images_path)
images_dir.mkdir(parents=True, exist_ok=True)
videos_dir = Path(videos_path)
videos_dir.mkdir(parents=True, exist_ok=True)
output_images_dir = Path(output_images_path)
output_images_dir.mkdir(parents=True, exist_ok=True)
output_videos_dir = Path(output_videos_path)
output_videos_dir.mkdir(parents=True, exist_ok=True)


def create_detection_net(config_path, weights_path):
    net = cv2.dnn_DetectionModel(config_path, weights_path)
    net.setInputSize(608, 608)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net


def get_processed_image(img, net, confThreshold, nmsThreshold):
    classes, confidences, boxes = net.detect(img, confThreshold, nmsThreshold)
    for cl, score, (left, top, width, height) in zip(classes, confidences, boxes):
        start_point = (int(left), int(top))
        end_point = (int(left + width), int(top + height))
        color = color_list[cl]
        img = cv2.rectangle(img, start_point, end_point, color, 2)  # draw class box
        text = f'{class_list[cl]}: {score:0.2f}'
        (test_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_ITALIC, 0.6, 1)
        end_point = (int(left + test_width + 2), int(top - text_height - 2))
        img = cv2.rectangle(img, start_point, end_point, color, -1)
        cv2.putText(img, text, start_point, cv2.FONT_ITALIC, 0.6, 0, 1)  # print class type with score
    return img


def detect_image(image_path, confThreshold=0.5, nmsThreshold=0.5):
    print("\nObject detection on " + Path(image_path).name)
    net = create_detection_net(configPath, weightsPath)
    img = cv2.imread(image_path)
    output_image_path = os.path.join(output_images_dir, Path(image_path).name)
    result_img = get_processed_image(img, net, confThreshold, nmsThreshold)
    cv2.imwrite(output_image_path, result_img)
    return output_image_path


def detect_video(video_path, confThreshold=0.5, nmsThreshold=0.5):
    print("\nObject detection on " + Path(video_path).name)
    net = create_detection_net(configPath, weightsPath)
    cap = cv2.VideoCapture(video_path)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    output_video_path = os.path.join(output_videos_dir, Path(video_path).name)
    out = cv2.VideoWriter(output_video_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), frame_size)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        frame = get_processed_image(frame, net, confThreshold, nmsThreshold)
        cv2.imshow(Path(video_path).name, frame)
        out.write(frame)
        if cv2.waitKey(int(50 // cap.get(cv2.CAP_PROP_FPS))) & 0xFF == 27:  # 27 = ESC ASCII code
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return output_video_path


def detect_webcam(confThreshold=0.5, nmsThreshold=0.5):
    net = create_detection_net(configPath, weightsPath)
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while cam.isOpened():
        ret, frame = cam.read()
        if ret is False:
            break
        frame = get_processed_image(frame, net, confThreshold, nmsThreshold)
        cv2.imshow('Webcam (Press ESC for exit)', frame)
        if cv2.waitKey(int(1000 // cam.get(cv2.CAP_PROP_FPS))) & 0xFF == 27:  # 27 = ESC ASCII code
            break
    cam.release()
    cv2.destroyAllWindows()


detect_webcam()

# img_path = get_file("cars.jpg", "https://github.com/sicara/tf2-yolov4/raw/master/notebooks/images/cars.jpg", cache_dir="images/", cache_subdir="")
# output_img_path = detect_image(img_path)

# vid_path = "videos/cars_small.mp4"
# output_vid_path = detect_video(vid_path)
