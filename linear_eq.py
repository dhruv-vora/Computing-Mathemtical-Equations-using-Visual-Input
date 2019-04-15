import math
# import time
from collections import deque

import cv2
import numpy as np

import digit_recognizer as dr
import time
def ans_print(frame, x, y):
    cv2.putText(frame, y, (5, 500), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, x, (5, 460), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


def lineq():
    number = []
    counter1 = 0
    coeff=[]
    multi=""
    counter = 0
    # def linearequ(frame, counter):



    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    lowergreen = np.array([50, 100, 50])
    uppergreen = np.array([90, 255, 255])
    center_points = deque()

    # the black board for the models
    board = np.zeros((230, 230), dtype='uint8')
    equ = ""

    while cap.isOpened():
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

        # rectangle frame for - operator
        cv2.rectangle(frame, (250, 50), (450, 250), (100, 100, 255), 2)
        roi2 = frame[50:250, 250:450, :]
        cv2.line(frame, (270, 150), (430, 150), (200, 200, 200), 10)

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # detecting colours in the range
        roi_range = cv2.inRange(hsv_roi, lowergreen, uppergreen)
        # applying contours on the detected colours
        contours, hierarchy = cv2.findContours(roi_range.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hsv_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
        roi_range2 = cv2.inRange(hsv_roi2, lowergreen, uppergreen)
        contours2, hierarchy2 = cv2.findContours(roi_range2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # the text to be displayed on the screen
        predict1_text = "Solving Linear Equation: "
        predict2_text = "Enter number and press to 's' to save: "
        numbers = []
        i=0
        # flags to check when drawing started and when stopped
        drawing_started = False
        drawing_stopped = False
        
        # detecting dot in - rectangle
        if len(contours2) > 0 and counter == 0:
            drawing_started = True
            #print("-")
            counter += 1

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

        # the board is resized for the prediction
        input = cv2.resize(board, (28, 28))
        # applying morphological transformation on the drawn digit
        if np.max(board) != 0 and drawing_started == True and drawing_stopped == True:
            kernel = (5, 5)
            input = cv2.morphologyEx(input, cv2.MORPH_OPEN, kernel)
            board = cv2.morphologyEx(board, cv2.MORPH_OPEN, kernel)
            drawing_started = False
            drawing_stopped = False
        # predicting the digit using LR and CNN
        if np.max(board) != 0:
            LR_input = input.reshape(1, 784)
            test_x = input.reshape((1, 28, 28, 1))
            # prediction1 = np.argmax(
            #LRmodel.predict(LR_input, dr.LR_params.item().get('weights'), dr.LR_params.item().get('base')))
            prediction2 = np.argmax(dr.model_conv.predict(test_x))

            #numbers.append(prediction2)
            predict2_text += str(prediction2)
        # displaying the text on the screen
        cv2.putText(frame, predict1_text, (5, 380), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, predict2_text, (5, 420), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

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
            if counter is 1:
                a=str(prediction2)
                b=int("-"+ a)
                number.append(b)
                counter = 0
            else:
                number.append(prediction2)
        elif k == ord('m'):
            print(multi)
            l=len(number)
            for i in range (0,l):
                multi+= str(number[i])
            print(multi)
            coeff.append(int(multi))
            #j=j+1
            number=[]
            multi=""
            counter=0        
        elif k == ord('e'):
            x1 = coeff[0]
            y1 = coeff[1]
            c1 = coeff[2]
            x2 = coeff[3]
            y2 = coeff[4]
            c2 = coeff[5]
            # x1 = 3
            # y1 = 4
            # c1 = 7
            # x2 = 4
            # y2 = 3
            # c2 = 7

            print(x1)
            print(y1)
            print(c1)
            print(x2)
            print(y2)
            print(c2)

            

            # linearequ(frame)
            x = "Value of x = "
            y = "Value of y = "
            equ1 = np.array([[x1, y1], [x2, y2]])
            equ2 = np.array([c1, c2])

            ans = np.linalg.solve(equ1, equ2)

            print(ans.item(0))
            print(ans.item(1))
            print(ans)
            x += str(ans.item(0))
            y += str(ans.item(1))
            counter1 = 1


            # time.sleep(10)
        if counter1 is 1:
            ans_print(frame, x, y)
            counter1 = 1
        # cv2.imshow('input', input)
        cv2.imshow('frame', frame)
        cv2.moveWindow("frame", 100, 20)
        # cv2.imshow('board', board)

    cap.release()
    cv2.destroyAllWindows()
# TODO DO quadratic