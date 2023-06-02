import cv2
import mediapipe as mp
import numpy as np
from scipy import spatial
from utils import moving_avg,match
from helpers import spatial_distance
import matplotlib.pyplot as plt
from PIL import Image

def register_user(name,signno):

    mp_hands = mp.solutions.hands

    handmodel = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    width, height = 1280, 720
    x1, y1 = 440, 260

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, width)
    cap.set(4, height)

    sign = []
    canvas = np.zeros((720, 1280, 3), np.uint8)
    xp, yp = 0, 0

    while True:

        cosine_sim_dict = dict()
        success, image = cap.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = handmodel.process(image)
        cv2.rectangle(image, (x1, y1), (x1+500, y1+200), (255, 0, 0), 1)

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            index_finger_tip = (int(
                hand_landmarks.landmark[8].x * width), int(hand_landmarks.landmark[8].y * height))
            thumb_tip = (int(
                hand_landmarks.landmark[4].x * width), int(hand_landmarks.landmark[4].y * height))
            (x_index, y_index) = index_finger_tip
            (x_thumb, y_thumb) = thumb_tip
            sd = spatial_distance(index_finger_tip, thumb_tip)
            x_pen, y_pen = (int((x_index+x_thumb)/2), int((y_index+y_thumb)/2))

            if sd < 70:
                if x_pen > x1 and x_pen < x1+500 and y_pen > y1 and y_pen < y1+200:
                    mode = 'w'
                    cv2.circle(img=image, center=(x_pen, y_pen),
                            radius=3, color=(0, 0, 230), thickness=-2)
                    if xp == 0 and yp == 0:
                        xp, yp = x_pen, y_pen

                    cv2.line(canvas,(xp,yp),(x_pen,y_pen),(255,255,255),2)
                    xp, yp = x_pen, y_pen
                    sign.append([x_pen/500,y_pen/200])
            else:
                mode = "dw"
                cv2.circle(img=image, center=(x_pen, y_pen),
                        radius=3, color=(0, 230, 0), thickness=-2)

        cv2.imshow("Webcam", cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_RGB2BGR))
        cv2.imshow("Canvas", cv2.cvtColor(cv2.flip(canvas[260:460,440:940], 1), cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(20)
        if key == ord('q'):
            #match sign
            sign = np.matrix(sign,np.double)
            sign_new = moving_avg(sign,5)

            # save sign
            np.save(f'database/{name}/strokes/sign{signno}.npy',sign_new)
            plt.plot(sign_new[:,0],sign_new[:,1])
            plt.axis('off')
            plt.savefig(f'database/{name}/images/sign{signno}.jpg',bbox_inches='tight',transparent=True, pad_inches=0)
            print("Quit")
            break

    cap.release()


def authenticate_user(name):
    mp_hands = mp.solutions.hands

    handmodel = mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    width, height = 1280, 720
    x1, y1 = 440, 260

    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    sign = []
    canvas = np.zeros((720, 1280, 3), np.uint8)
    xp, yp = 0, 0

    while True:

        cosine_sim_dict = dict()
        success, image = cap.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = handmodel.process(image)
        cv2.rectangle(image, (x1, y1), (x1+500, y1+200), (255, 0, 0), 1)

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            index_finger_tip = (int(
                hand_landmarks.landmark[8].x * width), int(hand_landmarks.landmark[8].y * height))
            thumb_tip = (int(
                hand_landmarks.landmark[4].x * width), int(hand_landmarks.landmark[4].y * height))
            (x_index, y_index) = index_finger_tip
            (x_thumb, y_thumb) = thumb_tip
            sd = spatial_distance(index_finger_tip, thumb_tip)
            x_pen, y_pen = (int((x_index+x_thumb)/2), int((y_index+y_thumb)/2))

            if sd < 70:
                if x_pen > x1 and x_pen < x1+500 and y_pen > y1 and y_pen < y1+200:
                    mode = 'w'
                    cv2.circle(img=image, center=(x_pen, y_pen),
                            radius=3, color=(0, 0, 230), thickness=-2)
                    if xp == 0 and yp == 0:
                        xp, yp = x_pen, y_pen

                    cv2.line(canvas,(xp,yp),(x_pen,y_pen),(255,255,255),2)
                    xp, yp = x_pen, y_pen
                    sign.append([x_pen/500,y_pen/200])
            else:
                mode = "dw"
                cv2.circle(img=image, center=(x_pen, y_pen),
                        radius=3, color=(0, 230, 0), thickness=-2)

        cv2.imshow("Webcam", cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_RGB2BGR))
        cv2.imshow("Canvas", cv2.cvtColor(cv2.flip(canvas[260:460,440:940], 1), cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(20)
        if key == ord('q'):
            #match sign
            sign = np.matrix(sign,np.double)
            sign_new = moving_avg(sign,5)
            plt.plot(sign_new[:,0],sign_new[:,1])
            plt.axis('off')
            plt.savefig('database/made/sign.jpg',bbox_inches='tight',transparent=True, pad_inches=0)
            imgref = np.asarray(Image.open('database/made/sign.jpg'))/255
            m = match(name,imgref,sign_new)
            if(m):
                print('Signature matched')
            else:
                print('Signature did not match')

            print("Quit")
            break

    cap.release()