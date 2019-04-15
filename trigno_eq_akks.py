import math
# import time
from collections import deque

import cv2
import numpy as np

import digit_recognizer as dr

def print_ans(frame, counter_aakash, text):
    print(text+"ans")
    ans = "Answer = " + text
    cv2.putText(frame, ans, (5, 645), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

def trigno():
    counter=0
    counter_aakash = 0
    angle=[]
    multi=""
    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    # collection of points to draw
    center_points = deque()

    # green colour pointer to be detected
    lowergreen = np.array([50, 100, 50])
    uppergreen = np.array([90, 255, 255])

    # the black board for the models
    board = np.zeros((230, 230), dtype='uint8')
    equ = ""

    while cap.isOpened():
        counter_aakash = 0
        ret, frame = cap.read()
        # flipping the frame
        frame = cv2.flip(frame, 1)
        # applying gaussian blur
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # drawing the rectangle for the board
        frame = cv2.resize(frame, (1280,720))
        # box area to detect digits
        cv2.rectangle(frame, (550, 50), (750, 250), (100, 100, 255), 2)
        roi = frame[50:250, 550:750, :]
        
        # rectangle frame for sin operator
        cv2.rectangle(frame, (150, 50), (275, 175), (100, 100, 255), 2)
        roi2 = frame[50:175, 150:275]
        cv2.putText(frame, "sin", (190, 112), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # rectangle frame for cos operator
        cv2.rectangle(frame, (150, 205), (275, 330), (100, 100, 255), 2)
        roi3 = frame[205:330, 150:275, :]
        cv2.putText(frame, "cos", (190, 267), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # rectangle frame for tan eqaution
        cv2.rectangle(frame, (150, 360), (275, 485), (100, 100, 255), 2)
        roi4 = frame[360:485, 150:275, :]
        cv2.putText(frame, "tan", (190, 422), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # cv2.rectangle(frame, (1050, 50), (1175, 175), (100, 100, 255), 2)
        # roi5 = frame[50:175, 1050:1175, :]
        # cv2.line(frame, (1112, 60), (1112, 165), (200, 200, 200), 10)
        # cv2.line(frame, (1060, 112), (1165, 112), (200, 200, 200), 10)
        #
        # # rectangle frame for - operator
        # cv2.rectangle(frame, (1050, 205), (1175, 330), (100, 100, 255), 2)
        # roi6 = frame[205:330, 1050:1175, :]
        # cv2.line(frame, (1060, 267), (1165, 267), (200, 200, 200), 10)
        #
        # # rectangle frame for * operator
        # cv2.rectangle(frame, (1050, 360), (1175, 485), (100, 100, 255), 2)
        # roi7 = frame[360:485, 1050:1175, :]
        # cv2.line(frame, (1112, 370), (1112, 475), (200, 200, 200), 10)
        # cv2.line(frame, (1060, 422), (1165, 422), (200, 200, 200), 10)
        # cv2.line(frame, (1070, 380), (1155, 465), (200, 200, 200), 10)
        # cv2.line(frame, (1155, 380), (1070, 465), (200, 200, 200), 10)
        #
        # # rectangle frame for / operator
        # cv2.rectangle(frame, (1050, 515), (1175, 640), (100, 100, 255), 2)
        # roi8 = frame[515:640, 1050:1175, :]
        # cv2.line(frame, (1155, 535), (1070, 620), (200, 200, 200), 10)
        #
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # detecting colours in the range
        roi_range = cv2.inRange(hsv_roi, lowergreen, uppergreen)
        # applying contours on the detected colours
        contours, hierarchy = cv2.findContours(roi_range.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hsv_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
        roi_range2 = cv2.inRange(hsv_roi2, lowergreen, uppergreen)
        contours2, hierarchy2 = cv2.findContours(roi_range2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hsv_roi3 = cv2.cvtColor(roi3, cv2.COLOR_BGR2HSV)
        roi_range3 = cv2.inRange(hsv_roi3, lowergreen, uppergreen)
        contours3, hierarchy3 = cv2.findContours(roi_range3.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hsv_roi4 = cv2.cvtColor(roi4, cv2.COLOR_BGR2HSV)
        roi_range4 = cv2.inRange(hsv_roi4, lowergreen, uppergreen)
        contours4, hierarchy4 = cv2.findContours(roi_range4.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # hsv_roi5 = cv2.cvtColor(roi5, cv2.COLOR_BGR2HSV)
        # roi_range5 = cv2.inRange(hsv_roi5, lowergreen, uppergreen)
        # contours5, hierarchy5 = cv2.findContours(roi_range5.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        # hsv_roi6 = cv2.cvtColor(roi6, cv2.COLOR_BGR2HSV)
        # roi_range6 = cv2.inRange(hsv_roi6, lowergreen, uppergreen)
        # contours6, hierarchy6 = cv2.findContours(roi_range6.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        # hsv_roi7 = cv2.cvtColor(roi7, cv2.COLOR_BGR2HSV)
        # roi_range7 = cv2.inRange(hsv_roi7, lowergreen, uppergreen)
        # contours7, hierarchy7 = cv2.findContours(roi_range7.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        # hsv_roi8 = cv2.cvtColor(roi8, cv2.COLOR_BGR2HSV)
        # roi_range8 = cv2.inRange(hsv_roi8, lowergreen, uppergreen)
        # contours8, hierarchy8 = cv2.findContours(roi_range8.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        # the text to be displayed on the screen
        predict1_text = "Solving Trignometric Equation: "
        predict2_text = "Enter number and press to 's' to save: "
        
        # flags to check when drawing started and when stopped
        drawing_started = False
        drawing_stopped = False
        if len(contours) > 0:
            drawing_started = True
            # getting max contours from the contours
            max_contours = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_contours)
            # to avoid divided by zero error
            try:
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            except:
                continue
            # center obtained is appended to the deque
            center_points.appendleft(center)
        else:
            drawing_stopped = False
        for i in range(1, len(center_points)):
            if math.sqrt((center_points[i - 1][0] - center_points[i][0]) ** 2 +
                         (center_points[i - 1][1] - center_points[i][1]) ** 2) < 50:
                cv2.line(roi, center_points[i - 1], center_points[i], (200, 200, 200), 5, cv2.LINE_AA)
                cv2.line(board, (center_points[i - 1][0] + 15, center_points[i - 1][1] + 15),
                         (center_points[i][0] + 15, center_points[i][1] + 15), 255, 7, cv2.LINE_AA)
        
        # detecting dot in sin rectangle        
        if len(contours2) > 0 and counter is 0:
            counter_aakash = 1
            print_ans(frame, counter_aakash, str(math.sin(ang)))
            drawing_started = True
            print(math.sin(ang))
            print("sin")

        # detecting dot in cos rectangle
        if len(contours3) > 0 and counter is 0:
            counter_aakash = 1
            print_ans(frame, counter_aakash, str(math.cos(ang)))
            drawing_started = True
            print(math.cos(ang))
            print("cos")

        # detecting dot in tan rectangle
        if len(contours4) > 0 and counter is 0:
            counter_aakash = 1
            print_ans(frame, counter_aakash, str(math.tan(ang)))
            drawing_started = True
            print(math.tan(ang))
            print("tan")
            
        # # detecting dot in + rectangle
        # if len(contours5) > 0 and counter is 0:
        #     drawing_started = True
        #     # print("/")
        #     # digits.append('/')
        #     # counter += 1
        #
        # # detecting dot in - rectangle
        # if len(contours6) > 0 and counter is 0:
        #     drawing_started = True
        #     # print("^")
        #     # digits.append('**')
        #     # counter += 1
        #
        # # detecting dot in * rectangle
        # if len(contours7) > 0:
        #     drawing_started = True
        #     # print("root")
        #
        # # detecting dot in / rectangle
        # if len(contours8) > 0:
        #     drawing_started = True
        #     # print("root")

        # the board is resized for the prediction
        input = cv2.resize(board, (28, 28))
        # applying morphological transformation on the drawn digit
        if np.max(board) != 0 and drawing_started is True and drawing_stopped is True:
            kernel = (5, 5)
            input = cv2.morphologyEx(input, cv2.MORPH_OPEN, kernel)
            board = cv2.morphologyEx(board, cv2.MORPH_OPEN, kernel)
            drawing_started = False
            drawing_stopped = False
        
        if np.max(board) != 0:
            LR_input = input.reshape(1, 784)
            test_x = input.reshape((1, 28, 28, 1))
            prediction2 = np.argmax(dr.model_conv.predict(test_x))

            #angle.append(prediction2)
            predict2_text += str(prediction2)
        
        # displaying the text on the screen
        cv2.putText(frame, predict1_text, (5, 555), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, predict2_text, (5, 595), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)



        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        # clearing the board
        elif k == ord('c'):
            board.fill(0)
            center_points.clear()
        elif k == ord('p'):
            p = not p
        elif k == ord('s'):
            angle.append(prediction2)
        elif k == ord('m'):
            print(multi)
            l=len(angle)
            for i in range (0,l):
                multi+= str(angle[i])
            print(multi)
            ang= int(multi)*((math.pi)/180)
            angle=[]
            multi=""
        elif k == ord('v'):
            counter_aakash = 0
            
        # cv2.imshow('input', input)
        cv2.imshow('frame', frame)
        cv2.moveWindow("frame", 100, 20)
        # cv2.imshow('board', board)
        
    cap.release()
    cv2.destroyAllWindows()
