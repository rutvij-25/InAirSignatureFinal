import streamlit as st
import os
from air_signature import register_user,authenticate_user
import cv2
import mediapipe as mp
import numpy as np
from scipy import spatial
from utils import moving_avg,match
from helpers import spatial_distance
import matplotlib.pyplot as plt
from PIL import Image

if 'sign' not in st.session_state:
    st.session_state['sign'] = []

def main():
    st.sidebar.title("In-air Signature System")
    app_mode = st.sidebar.selectbox("Choose your option", ["Register", "Authenticate"])

    if app_mode == "Register":
        register()
    elif app_mode == "Authenticate":
        authenticate()

def register():
    st.title("Registration Page")
    user_input = st.text_input("Enter your name")
    register = st.button("Register")
    
    if register and user_input:
        
        if os.path.exists(f'database/{user_input}'):
            st.write('User already exists')
        else:
            os.mkdir(f'database/{user_input}')
            os.mkdir(f'database/{user_input}/images')
            os.mkdir(f'database/{user_input}/strokes')
            st.write('User registered successfully, please sign')

    checkbox_state = st.checkbox('Show camera')
    options = ['Sign1','Sign2','Sign3']
    selected_option = st.selectbox('Select an option', options)
    stopbutton = st.button("Stop")
    if checkbox_state:
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

        canvas = np.zeros((720, 1280, 3), np.uint8)
        xp, yp = 0, 0

        video_placeholder = st.empty()
        video_placeholder2 = st.empty()

        while True:

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
                        st.session_state['sign'].append([x_pen/500,y_pen/200])
                else:
                    mode = "dw"
                    cv2.circle(img=image, center=(x_pen, y_pen),
                            radius=3, color=(0, 230, 0), thickness=-2)
            video_placeholder.image(cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_RGB2BGR), channels="BGR")
            video_placeholder2.image(cv2.cvtColor(cv2.flip(canvas[260:460,440:940], 1), cv2.COLOR_RGB2BGR), channels="BGR")
            if selected_option == 'Sign1' and user_input and stopbutton:
                s = np.matrix(st.session_state['sign'],np.double)
                s = moving_avg(s,5)
                np.save(f'database/{user_input}/strokes/sign1.npy',s)
                plt.plot(s[:,0],s[:,1])
                plt.axis('off')
                plt.savefig(f'database/{user_input}/images/sign1.jpg',bbox_inches='tight',transparent=True, pad_inches=0)
                checkbox_state = False
                st.session_state['sign'] = []
                st.write('Sign 1 stored successfully')
                video_placeholder2.image(cv2.cvtColor(cv2.flip(canvas[260:460,440:940], 1), cv2.COLOR_RGB2BGR), channels="BGR")
                break
            
            if selected_option == 'Sign2' and user_input and stopbutton:

                s = np.matrix(st.session_state['sign'],np.double)
                s = moving_avg(s,5)
                np.save(f'database/{user_input}/strokes/sign2.npy',s)
                plt.plot(s[:,0],s[:,1])
                plt.axis('off')
                plt.savefig(f'database/{user_input}/images/sign2.jpg',bbox_inches='tight',transparent=True, pad_inches=0)
                checkbox_state = False
                st.session_state['sign'] = []
                st.write('Sign 2 stored successfully')
                break

            if selected_option == 'Sign3' and user_input and stopbutton:
        
                s = np.matrix(st.session_state['sign'],np.double)
                s = moving_avg(s,5)
                np.save(f'database/{user_input}/strokes/sign3.npy',s)
                plt.plot(s[:,0],s[:,1])
                plt.axis('off')
                plt.savefig(f'database/{user_input}/images/sign3.jpg',bbox_inches='tight',transparent=True, pad_inches=0)
                checkbox_state = False
                st.session_state['sign'] = []
                st.write('Sign 3 stored successfully')
                break
            
        
   


def authenticate():
    
    st.title("Authentication Page")
    user_input = st.text_input("Enter your name")
    login = st.button("Login")

    if user_input and login:
        if os.path.exists(f'database/{user_input}'):
            st.write('User exists, please sign')
        else:
            st.write('User does not exist')

    authenticate = st.button("Authenticate")


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

    video_placeholder = st.empty()
    video_placeholder2 = st.empty()

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
                    st.session_state['sign'].append([x_pen/500,y_pen/200])
            else:
                mode = "dw"
                cv2.circle(img=image, center=(x_pen, y_pen),
                        radius=3, color=(0, 230, 0), thickness=-2)
        video_placeholder.image(cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_RGB2BGR), channels="BGR")
        video_placeholder2.image(cv2.cvtColor(cv2.flip(canvas[260:460,440:940], 1), cv2.COLOR_RGB2BGR), channels="BGR")
    
        
        if authenticate and user_input:
            sign = np.matrix(st.session_state['sign'],np.double)
            sign_new = moving_avg(sign,5)
            np.save(f'database/made/sign.npy',sign_new)
            plt.plot(sign_new[:,0],sign_new[:,1])
            plt.axis('off')
            plt.savefig(f'database/made/sign.jpg',bbox_inches='tight',transparent=True, pad_inches=0)
            imgref = np.asarray(Image.open('database/made/sign.jpg'))/255
            m = match(user_input,imgref,sign_new)
            video_placeholder = st.empty()
            video_placeholder2 = st.empty()

            if(m):
                st.write('Signature matched')
            else:
                st.write('Signature did not match')
            break


if __name__ == "__main__":
    main()
