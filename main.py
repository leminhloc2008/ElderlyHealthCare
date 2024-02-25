import cv2
import torch
from ultralytics import YOLO
import time
import threading
import queue
import math
import numpy as np
import imutils
import telepot
import os
from deepface import DeepFace
import serial

from datetime import datetime
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from scipy.spatial.distance import pdist, squareform, cdist

device = "0" if torch.cuda.is_available() else "cpu"
if device == "0":
    torch.cuda.set_device(0)


class PosePredictor(DetectionPredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'pose'

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes,
                                        nc=len(self.model.names))

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            shape = orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(
                Results(orig_img=orig_img,
                        path=img_path,
                        names=self.model.names,
                        boxes=pred[:, :6],
                        keypoints=pred_kpts))
        return results

# Load YOLOV8 model
model = YOLO('yolo\yolov8m-pose.pt')
objectModel = YOLO('yolo\yolov8m.pt')
HARmodel = YOLO('model\HAR.pt')
print(model.device.type)
results = model("sample.jpg")
print(model.device.type)

name = HARmodel.names

# Kết nối đến arduino
ser = serial.Serial('COM4', 9600)

predictor = PosePredictor(overrides=dict(model='yolo\yolov8m-pose.pt'))

token = '6591943273:AAGp1NNT_GK3RgJ9E81XXP1zpUAaDYQREA4'
receiver_id = 5606318609

bot = telepot.Bot(token)

# Load DEEPFACE model
faceModel = DeepFace.build_model('Emotion')

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

class VideoCapture:

    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def release(self):
        self.cap.release()

def angle_between(vector1, vector2):
    vector1_norm = vector1 / np.linalg.norm(vector1)
    vector2_norm = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(vector1_norm, vector2_norm)
    angle = np.arccos(dot_product)
    return angle

scale_w = 1.2 / 2
scale_h = 4 / 2

SOLID_BACK_COLOR = (41, 41, 41)

mouse_pts = []

def get_mouse_points(event, x, y, flags, param):
    global mouseX, mouseY, mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        cv2.circle(image, (x, y), 3, (224, 224, 224), cv2.FILLED)
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))

def get_camera_perspective(img, src_points):
    IMAGE_H = img.shape[0]
    IMAGE_W = img.shape[1]
    src = np.float32(np.array(src_points))
    dst = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    return M, M_inv

def plot_points_on_bird_eye_view(frame, pedestrian_boxes, M, scale_w, scale_h):
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]

    node_radius = 10
    color_node = (192, 133, 156)
    thickness_node = 20
    solid_back_color = (41, 41, 41)

    blank_image = np.zeros(
        (int(frame_h * scale_h), int(frame_w * scale_w), 3), np.uint8
    )
    blank_image[:] = solid_back_color
    warped_pts = []
    for i in range(len(pedestrian_boxes)):

        mid_point_x = int(
            (pedestrian_boxes[i][1] * frame_w + pedestrian_boxes[i][3] * frame_w) / 2
        )
        mid_point_y = int(
            (pedestrian_boxes[i][0] * frame_h + pedestrian_boxes[i][2] * frame_h) / 2
        )

        pts = np.array([[[mid_point_x, mid_point_y]]], dtype="float32")
        warped_pt = cv2.perspectiveTransform(pts, M)[0][0]
        warped_pt_scaled = [int(warped_pt[0] * scale_w), int(warped_pt[1] * scale_h)]

        warped_pts.append(warped_pt_scaled)
        bird_image = cv2.circle(
            blank_image,
            (warped_pt_scaled[0], warped_pt_scaled[1]),
            node_radius,
            color_node,
            thickness_node,
        )

    return warped_pts, bird_image

image = None

def start_detect(cap):

    FT = {}
    headCenterx = {}
    headCentery = {}
    u = {}
    v = {}
    writer = None

    isTurnedOn = False

    SOLID_BACK_COLOR = (41, 41, 41)

    num_mouse_points = 0
    frame_num = 0
    time_count = 0
    status = "Bình Thường"

    emotion_translations = {
        "angry": "Giận Dữ",
        "disgust": "Khó Chịu",
        "fear": "Sợ Hãi",
        "happy": "Hạnh Phúc",
        "neutral": "Trung Lập",
        "sad": "Buồn",
        "surprise": "Ngạc Nhiên",
    }

    activity_translations = {
        "": "Khong xac dinh",
        "Drinking": "Uong Nuoc",
        "Fall_down": "Nga",
        "Lying_down": "Dang Nam",
        "Nearly_fall": "Sap Nga",
        "Walking": "Di bo",
        "Standing": "Dung",
        "Walking_on_Stairs": "Di cau thang",
        "Sitting": "Ngoi"
    }
    print("cam: " + str(isIpCam))
    while True:
        frame_num += 1
        time_count += 1
        if isIpCam:
            frame = cap.read()
        else:
            ret,frame = cap.read()
        frame = imutils.resize(frame, width=700)
        height, width, channels = frame.shape
        frame_h = frame.shape[0]
        frame_w = frame.shape[1]
        start_time = time.time()

        if not isIpCam:
            if not ret:
                break

        if frame_num == 1:
            while True:
                image = frame
                cv2.imshow("Chon diem", image)
                cv2.waitKey(1)
                if len(mouse_pts) == 7:
                    cv2.destroyWindow("Chon diem")
                    break

                # vẽ đoạn
                if len(mouse_pts) == 2:
                    cv2.line(image, mouse_pts[0], mouse_pts[1], (224, 224, 224), 2)

                if len(mouse_pts) == 3:
                    cv2.line(image, mouse_pts[0], mouse_pts[2], (224, 224, 224), 2)

                if len(mouse_pts) == 4:
                    cv2.line(image, mouse_pts[2], mouse_pts[3], (224, 224, 224), 2)
                    cv2.line(image, mouse_pts[3], mouse_pts[1], (224, 224, 224), 2)
                if len(mouse_pts) == 6:
                    cv2.line(image, mouse_pts[4], mouse_pts[5], (224, 224, 224), 2)

                first_frame_display = False
            four_points = mouse_pts

            # Get perspective
            M, Minv = get_camera_perspective(frame, four_points[0:4])
            pts = src = np.float32(np.array([four_points[4:]]))
            warped_pt = cv2.perspectiveTransform(pts, M)[0]
            d_thresh = np.sqrt(
                (warped_pt[0][0] - warped_pt[1][0]) ** 2
                + (warped_pt[0][1] - warped_pt[1][1]) ** 2
            )

            pedestrian_detect = frame

        pts = np.array(
            [four_points[0], four_points[1], four_points[3], four_points[2]], np.int32
        )

        # Chạy pose detection trên khung hình
        results1 = predictor(frame)
        results = objectModel.track(frame, persist=True, classes=[59])
        resultsHar = HARmodel.predict(frame)
        # Hiển thị
        frame_ = frame
        names = objectModel.names
        bedBoxes = []
        if len(results[0]) > 0:
            for r in results:
                bboxes = r.boxes.xyxy
                bedCenterX = (bboxes[0][0].item() + bboxes[0][2].item()) / 2
                bedCenterY = (bboxes[0][1].item() + bboxes[0][3].item()) / 2
                bedCoor = (int(bboxes[0][1].item()), int(bboxes[0][0].item()), int(bboxes[0][3].item()), int(bboxes[0][2].item()))
                cv2.circle(frame_, (int(bedCenterX), int(bedCenterY)), 3, (0, 0, 255), cv2.FILLED)
                bedBoxes.append(bedCoor)

            warped_ptsBed, bird_image = plot_points_on_bird_eye_view(
                frame, bedBoxes, M, scale_w, scale_h
            )

        if len(results1[0].keypoints) > 0:
            for pos in range(0, len(results1[0].keypoints)):
                box = results1[0].boxes[pos].xyxy
                classes = ""
                if len(resultsHar[0]) > pos:
                    classes = name[int(resultsHar[0].boxes[pos].cls)]
                for idx, kpt in enumerate(results1[0].keypoints[pos]):
                    x = int(float(kpt[0]))
                    y = int(float(kpt[1]))

                    if idx == 1:
                        left_eye_x = x
                        left_eye_y = y
                    if idx == 3:
                        left_ear_x = x
                        left_ear_y = y
                    if idx == 5:
                        left_shoulder_x = x
                        left_shoulder_y = y
                    if idx == 7:
                        left_arm_x = x
                        left_arm_y = y
                    if idx == 9:
                        left_hand_x = x
                        left_hand_y = y
                    if idx == 11:
                        left_hip_x = x
                        left_hip_y = y
                    if idx == 13:
                        left_leg_x = x
                        left_leg_y = y
                    if idx == 15:
                        left_foot_x = x
                        left_foot_y = y
                    if idx == 2:
                        right_eye_x = x
                        right_eye_y = y
                    if idx == 4:
                        right_ear_x = x
                        right_ear_y = y
                    if idx == 6:
                        right_shoulder_x = x
                        right_shoulder_y = y
                    if idx == 8:
                        right_arm_x = x
                        right_arm_y = y
                    if idx == 10:
                        right_hand_x = x
                        right_hand_y = y
                    if idx == 12:
                        right_hip_x = x
                        right_hip_y = y
                    if idx == 14:
                        right_leg_x = x
                        right_leg_y = y
                    if idx == 16:
                        right_foot_x = x
                        right_foot_y = y

                    '''print(f"Keypoint {idx}: ({kpt[0]}, {kpt})")
                    annotated_frame = cv2.putText(annotated_frame, f"{idx}:({x}, {y})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                    annotated_frame = cv2.putText(annotated_frame, f"{idx}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)'''

                len_factor = math.sqrt(((left_shoulder_y - left_hip_y) ** 2 + (left_shoulder_x - left_hip_x) ** 2))
                first = (right_hip_x + left_hip_x) / 2 - (right_shoulder_x + left_shoulder_x) / 2
                second = (right_hip_y + left_hip_y) / 2 - (right_shoulder_y + left_shoulder_y) / 2
                if FT.get(pos) is None:
                    FT[pos] = math.sqrt(first * first + second * second)
                    u[pos] = (right_hip_x + left_hip_x) / 2 - (right_shoulder_x + left_shoulder_x) / 2
                    v[pos] = (right_hip_y + left_hip_y) / 2 - (right_shoulder_y + left_shoulder_y) / 2

                ST = math.sqrt(first * first + second * second)

                cosStFt = ((first * u[pos]) + (second * v[pos])) / (ST * FT[pos])
                angle = np.degrees(np.arccos(cosStFt))

                '''print(angle)
                print(str(first) + ' ' + str(second))
                print(str(u[pos]) + ' ' + str(v[pos]) + ' ' + str(first) + ' ' + str(second))
                print(str(FT[pos]) + ' ' + str(ST) + ' ' + str(DT) + ' ' + str(val))'''

                xmax = max(left_eye_x, left_ear_x, left_shoulder_x, left_arm_x, left_hand_x, left_hip_x, left_leg_x,
                           right_eye_x, right_ear_x, right_shoulder_x, right_arm_x, right_hand_x, right_hip_x, right_leg_x,
                           right_foot_x)
                xmin = min(left_eye_x, left_ear_x, left_shoulder_x, left_arm_x, left_hand_x, left_hip_x, left_leg_x,
                           right_eye_x, right_ear_x, right_shoulder_x, right_arm_x, right_hand_x, right_hip_x, right_leg_x,
                           right_foot_x)
                ymax = max(left_eye_y, left_ear_y, left_shoulder_y, left_arm_y, left_hand_y, left_hip_y, left_leg_y,
                           right_eye_y, right_ear_y, right_shoulder_y, right_arm_y, right_hand_y, right_hip_y, right_leg_y,
                           right_foot_y)
                ymin = min(left_eye_y, left_ear_y, left_shoulder_y, left_arm_y, left_hand_y, left_hip_y, left_leg_y,
                           right_eye_y, right_ear_y, right_shoulder_y, right_arm_y, right_hand_y, right_hip_y, right_leg_y,
                           right_foot_y)

                headCenterx[pos] = (left_ear_x + left_eye_x + right_eye_x + right_ear_x) / 4
                headCentery[pos] = (left_ear_y + left_eye_y + right_eye_y + right_ear_y) / 4

                dx = int(xmax) - int(xmin)
                dy = int(ymax) - int(ymin)
                difference = dy - dx
                print(difference)

                center_x = (xmax + xmin) / 2
                center_y = (ymax + ymin) / 2

                humanBoxes = []
                humanCoor = (int(box[0][1].item()), int(box[0][0].item()), int(box[0][3].item()), int(box[0][2].item()))
                humanBoxes.append(humanCoor)

                warped_ptsHuman, bird_image_ = plot_points_on_bird_eye_view(
                    frame, humanBoxes, M, scale_w, scale_h
                )

                cv2.circle(frame_, (int(center_x), int(center_y)), 3, (0, 0, 255), cv2.FILLED)

                if (left_shoulder_y > left_foot_y - len_factor and left_hip_y > left_foot_y - (
                        len_factor / 2) and left_shoulder_y > left_hip_y - (len_factor / 2)) or (
                        right_shoulder_y > right_foot_y - len_factor and right_hip_y > right_foot_y - (
                        len_factor / 2) and right_shoulder_y > right_hip_y - (len_factor / 2)) \
                        or difference < 0 or angle > 100 or str(classes) == 'Fall_down' or str(classes) == 'Nearly_fall':

                    print("check "+str(angle)+' '+str(pos))

                    if len(bedBoxes) > 0:
                        p = np.array(warped_ptsBed)
                        h = np.array(warped_ptsHuman)
                        print("Check")
                        print(warped_ptsBed)
                        print(warped_ptsHuman)
                        print("Stop Check")
                        dist_condensed = cdist(p, h)
                        color_10 = (80, 172, 110)
                        lineThickness = 4

                        for i in range(0, dist_condensed.shape[0]):
                            for j in range(0, dist_condensed.shape[1]):
                                print("dist "+str(dist_condensed.shape[0])+" "+str(dist_condensed.shape[1])+" "+str(dist_condensed[i,j])+ " "+str(d_thresh))
                                if dist_condensed[i, j] < d_thresh / 6 * 3:
                                    print("0")
                                    cv2.rectangle(frame_, (int(humanBoxes[0][1]), int(humanBoxes[0][0])), (int(humanBoxes[0][3]), int(humanBoxes[0][2])), (255, 128, 0), 2)
                                    cv2.putText(frame_, 'Di Ngu' + ' | ' + str(activity_translations[classes]), (int(humanBoxes[0][1]), int(humanBoxes[0][0]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
                                    '''bot.sendMessage(receiver_id, 'Người cần giám sát đã đi ngủ')
                                    filename = "D:\\Documents\\ElderlyHealthCare\\photo\\sendImage.jpg"
                                    cv2.imwrite(filename, frame_)
                                    bot.sendPhoto(receiver_id, photo=open(filename, 'rb'))
                                    os.remove(filename)
                                    status = "Đang Ngủ"'''
                                    if isTurnedOn:
                                        ser.write(b'L')
                                        isTurnedOn = False
                                else:
                                    print("1")
                                    cv2.rectangle(frame_, (int(humanBoxes[0][1]), int(humanBoxes[0][0])), (int(humanBoxes[0][3]), int(humanBoxes[0][2])), (0, 0, 255), 2)
                                    cv2.putText(frame_, 'Nga' + ' | ' + str(activity_translations[classes]), (int(humanBoxes[0][1]), int(humanBoxes[0][0]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                                    '''bot.sendMessage(receiver_id, 'Cảnh Báo: Phát hiện có người ngã')
                                    filename = "D:\\Documents\\ElderlyHealthCare\\photo\\sendImage.jpg"
                                    cv2.imwrite(filename, frame_)
                                    bot.sendPhoto(receiver_id, photo=open(filename, 'rb'))
                                    os.remove(filename)
                                    status = "Ngã"'''
                                    if not isTurnedOn:
                                        ser.write(b'H')
                                        isTurnedOn = True
                    else:
                        cv2.rectangle(frame_, (int(box[0][0].item()), int(box[0][1].item())), (int(box[0][2].item()), int(box[0][3].item())), (0, 0, 255), 2)
                        cv2.putText(frame_, 'Nga' + ' | ' + str(activity_translations[classes]), (int(box[0][0].item()), int(box[0][1].item()) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        print("1")
                        if not isTurnedOn:
                            ser.write(b'H')
                            isTurnedOn = True
                        '''bot.sendMessage(receiver_id, 'Cảnh Báo: Phát hiện có người ngã')
                        filename = "D:\\Documents\\ElderlyHealthCare\\photo\\sendImage.jpg"
                        cv2.imwrite(filename, frame_)
                        bot.sendPhoto(receiver_id, photo=open(filename, 'rb'))
                        os.remove(filename)
                        status = "Ngã"'''
                else:
                    cv2.rectangle(frame_, (int(box[0][0].item()), int(box[0][1].item())), (int(box[0][2].item()), int(box[0][3].item())), (0, 255, 0), 2)
                    cv2.putText(frame_, 'Binh Thuong' + ' | ' + str(activity_translations[classes]), (int(box[0][0].item()), int(box[0][1].item()) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
                    print("0")
                    if isTurnedOn:
                        ser.write(b'L')
                        isTurnedOn = False
                    status = "Bình Thường"

                # Gửi tin nhắn

                # Đổi kích thước khung hình
                resized_frame = cv2.resize(frame, (48, 48), interpolation=cv2.INTER_AREA)

                # Đổi màu khung hình
                gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

                # Xử lý ảnh cho DEEPFACE
                img = gray_frame.astype('float32') / 255.0
                img = np.expand_dims(img, axis=-1)
                img = np.expand_dims(img, axis=0)

                # Nhận diện cảm xúc bằng Deep Face
                preds = faceModel.predict(img)
                emotion_idx = np.argmax(preds)
                emotion = emotion_labels[emotion_idx]

                if time_count == 15:
                    '''bot.sendMessage(receiver_id, 'Đây là tin nhắn tự động, được thiết lập để gửi sau ' + str(time_count) + ' giây')
                    bot.sendMessage(receiver_id, 'Cảm xúc hiện tại: ' + str(emotion_translations[emotion]))
                    bot.sendMessage(receiver_id, 'Hành động hiện tại: ' + str(activity_translations[classes]))
                    filename = "D:\\Documents\\ElderlyHealthCare\\photo\\sendImage.jpg"
                    cv2.imwrite(filename, frame_)
                    bot.sendPhoto(receiver_id, photo=open(filename, 'rb'))
                    os.remove(filename)
                    time_count = 0'''

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print("FPS :", fps)

        cv2.putText(frame_, "FPS :"+str(int(fps)), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 0, 255), 1, cv2.LINE_AA)

        # Hiển thị
        cv2.imshow("Fall Detection", frame_)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Lưu video
        if writer is None:
            date_time = datetime.now().strftime("%m-%d-%Y %H-%M-%S")
            OUTPUT_PATH = 'output\output {}.avi'.format(date_time)
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30, (frame_.shape[1], frame_.shape[0]), True)
        writer.write(frame_)

        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break

    writer.release()

def webcam_detect():
    cv2.namedWindow("Chon diem")
    cv2.setMouseCallback("Chon diem", get_mouse_points)
    cap = cv2.VideoCapture(0)
    start_detect(cap)

def start_video(video_path):
    cv2.namedWindow("Chon diem")
    cv2.setMouseCallback("Chon diem", get_mouse_points)
    cap = cv2.VideoCapture(video_path)
    start_detect(cap)

isIpCam = False
def streamCam(streamUrl):
    global isIpCam
    isIpCam = True
    cv2.namedWindow("Chon diem")
    cv2.setMouseCallback("Chon diem", get_mouse_points)
    cap = VideoCapture(streamUrl)
    start_detect(cap)

def passCam(username, password):
    cv2.namedWindow("Chon diem")
    cv2.setMouseCallback("Chon diem", get_mouse_points)
    cap = cv2.VideoCapture("rtsp://" + username + ':' + password + "@192.168.1.64/1")
    start_detect(cap)

if __name__ == '__main__':
    cv2.namedWindow("Chon diem")
    cv2.setMouseCallback("Chon diem", get_mouse_points)
    webcam = False
    video_play = True
    ser.write(b'L')
    # H: bật đèn, L: tắt đèn
    if webcam:
        webcam_detect()
    if video_play:
        start_video('test\GFOD.mp4')

    cv2.destroyWindow()