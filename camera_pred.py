import math
import time
from collections import deque

import cv2
import numpy as np

import digit_recognizer as dr
# import linear_eq as lq
# import quadra_eq as qd
# import trigno_eq_akks as tr


def printeq(frame, text):
    cv2.putText(frame, text, (350, 340), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #time.sleep(2)

def printans(frame, text):
    # print("Hello :" + hello)
    cv2.putText(frame, text, (350, 380), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


def calci_main():
    digits = []
    counter = 0
    ans = 0
    counter_eq = 0
    counter3 = 0
    counter_akks = 0

    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    # collection of points to draw
    center_points = deque()
    center_points2 = deque()
    center_points3 = deque()
    center_points4 = deque()
    center_points5 = deque()
    # green colour pointer to be detected
    lowergreen = np.array([50, 100, 50])
    uppergreen = np.array([90, 255, 255])

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
        frame = cv2.resize(frame, (1280, 720))
        # box area to detect digits
        cv2.rectangle(frame, (550, 50), (750, 250), (100, 100, 255), 2)
        roi = frame[50:250, 550:750, :]

        # rectangle frame for + operator
        cv2.rectangle(frame, (1050, 50), (1175, 175), (100, 100, 255), 2)
        roi2 = frame[50:175, 1050:1175, :]
        cv2.line(frame, (1112, 60), (1112, 165), (200, 200, 200), 10)
        cv2.line(frame, (1060, 112), (1165, 112), (200, 200, 200), 10)

        # rectangle frame for - operator
        cv2.rectangle(frame, (1050, 205), (1175, 330), (100, 100, 255), 2)
        roi3 = frame[205:330, 1050:1175, :]
        cv2.line(frame, (1060, 267), (1165, 267), (200, 200, 200), 10)

        # rectangle frame for * operator
        cv2.rectangle(frame, (1050, 360), (1175, 485), (100, 100, 255), 2)
        roi4 = frame[360:485, 1050:1175, :]
        cv2.line(frame, (1112, 370), (1112, 475), (200, 200, 200), 10)
        cv2.line(frame, (1060, 422), (1165, 422), (200, 200, 200), 10)
        cv2.line(frame, (1070, 380), (1155, 465), (200, 200, 200), 10)
        cv2.line(frame, (1155, 380), (1070, 465), (200, 200, 200), 10)

        # rectangle frame for / operator
        cv2.rectangle(frame, (1050, 515), (1175, 640), (100, 100, 255), 2)
        roi5 = frame[515:640, 1050:1175, :]
        cv2.line(frame, (1155, 535), (1070, 620), (200, 200, 200), 10)

        # rectangle frame for ^ operator
        cv2.rectangle(frame, (150, 50), (275, 175), (100, 100, 255), 2)
        roi6 = frame[50:175, 150:275]
        cv2.line(frame, (170, 112), (212, 60), (200, 200, 200), 10)
        cv2.line(frame, (212, 60), (255, 112), (200, 200, 200), 10)

        # # rectangle frame for square root operator
        # cv2.rectangle(frame, (150, 205), (275, 330), (100, 100, 255), 2)
        # roi7 = frame[205:330, 150:275, :]
        # cv2.line(frame, (160, 267), (182, 310), (200, 200, 200), 10)
        # cv2.line(frame, (182, 310), (212, 225), (200, 200, 200), 10)
        # cv2.line(frame, (212, 225), (255, 225), (200, 200, 200), 10)
        #
        # # rectangle frame for linear eqaution
        # cv2.rectangle(frame, (150, 360), (275, 485), (100, 100, 255), 2)
        # roi8 = frame[360:485, 150:275, :]
        # cv2.putText(frame, "Linear", (170, 400), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, "Equation", (160, 450), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        #
        # # rectangle frame for quadratic equation
        # cv2.rectangle(frame, (150, 515), (275, 640), (100, 100, 255), 2)
        # roi9 = frame[515:640, 150:275, :]
        # cv2.putText(frame, "Quadratic", (150, 555), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, "Equation", (160, 600), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        #
        # # rectangle frame for trignometric equation
        # cv2.rectangle(frame, (325, 515), (450, 640), (100, 100, 255), 2)
        # roi10 = frame[515:640, 325:450, :]
        # cv2.putText(frame, "Trignometric", (325, 555), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, "Equation", (350, 600), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # cv2.rectangle(frame, (500, 600), (400, 600), (100, 100, 255), 2)

        # rectangle frame for ^ operator
        #    cv2.rectangle(frame, (400, 500), (600, 750), (100, 100, 255), 2)
        #    roi5 = frame[500:750, 500:600, :]

        # plus sign
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

        hsv_roi5 = cv2.cvtColor(roi5, cv2.COLOR_BGR2HSV)
        roi_range5 = cv2.inRange(hsv_roi5, lowergreen, uppergreen)
        contours5, hierarchy5 = cv2.findContours(roi_range5.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        hsv_roi6 = cv2.cvtColor(roi6, cv2.COLOR_BGR2HSV)
        roi_range6 = cv2.inRange(hsv_roi6, lowergreen, uppergreen)
        contours6, hierarchy6 = cv2.findContours(roi_range6.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # hsv_roi7 = cv2.cvtColor(roi7, cv2.COLOR_BGR2HSV)
        # roi_range7 = cv2.inRange(hsv_roi7, lowergreen, uppergreen)
        # contours7, hierarchy7 = cv2.findContours(roi_range7.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        # hsv_roi8 = cv2.cvtColor(roi8, cv2.COLOR_BGR2HSV)
        # roi_range8 = cv2.inRange(hsv_roi8, lowergreen, uppergreen)
        # contours8, hierarchy8 = cv2.findContours(roi_range8.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        # hsv_roi9 = cv2.cvtColor(roi9, cv2.COLOR_BGR2HSV)
        # roi_range9 = cv2.inRange(hsv_roi9, lowergreen, uppergreen)
        # contours9, hierarchy9 = cv2.findContours(roi_range9.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        # hsv_roi10 = cv2.cvtColor(roi10, cv2.COLOR_BGR2HSV)
        # roi_range10 = cv2.inRange(hsv_roi10, lowergreen, uppergreen)
        # contours10, hierarchy10 = cv2.findContours(roi_range10.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        predict1_text = "Solving Linear Equation: "
        predict2_text = "Enter number and press to 's' to save: "

        # the text to be displayed on the screen
        predict1_text = "Logistic Regression : "
        predict2_text = "Number : "
        predict3_text = "Equation : "
        predict4_text = "Answer : "

        numbers = []
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

        # detecting dot in + rectangle
        if len(contours2) > 0 and counter is 0:
            drawing_started = True
            digits.append('+')
            counter += 1
            # print("+")
            # print(type(contours2))
            # getting max contours from the contours
            # max_contours2 = max(contours2, key2=cv2.contourArea)
            # M = cv2.moments(max_contours2)
            # to avoid divided by zero error
            # try:
            # center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            # except:
            # continue
            # center obtained is appended to the deque
            # center_points2.appendleft(center)
            #    else:
            #        drawing_stopped2 = False
            #    for i in range(1, len(center_points2)):
            #        if math.sqrt((center_points2[i-1][0] - center_points2[i][0])**2 +
            #                     (center_points2[i-1][1] - center_points2[i][1])**2) < 50:
            #            cv2.line(roi2, center_points2[i-1], center_points2[i], (200, 200, 200), 5, cv2.LINE_AA)
            #            cv2.line(board2, (center_points2[i-1][0]+15, center_points2[i-1][1]+15),
            #                     (center_points2[i][0]+15, center_points2[i][1]+15), 255, 7, cv2.LINE_AA)

        # detecting dot in - rectangle
        if len(contours3) > 0 and counter is 0:
            drawing_started = True
            # print("-")
            digits.append('-')
            counter += 1

        # detecting dot in * rectangle
        if len(contours4) > 0 and counter is 0:
            drawing_started = True
            # print("*")
            digits.append('*')
            counter += 1

        # detecting dot in / rectangle
        if len(contours5) > 0 and counter is 0:
            drawing_started = True
            # print("/")
            digits.append('/')
            counter += 1

        # detecting dot in ^  rectangle
        if len(contours6) > 0 and counter is 0:
            drawing_started = True
            # print("^")
            digits.append('**')
            counter += 1

        # # detecting dot in root  rectangle
        # if len(contours7) > 0:
        #     drawing_started = True
        #     print("root")
        #
        # # detecting dot in linear  rectangle
        # if len(contours8) > 0:
        #     drawing_started = True
        #     # print("linear")
        #
        #     lq.lineq()
        #
        # # detecting dot in quadratic rectangle
        # if len(contours9) > 0:
        #     drawing_started = True
        #     # print("quadratic")
        #
        #     qd.quadr()
        #
        # # detecting dot in trignometric rectangle
        # if len(contours10) > 0:
        #     drawing_started = True
        #     # print("quadratic")
        #
        #     tr.trigno()

        # the board is resized for the prediction
        input = cv2.resize(board, (28, 28))
        # applying morphological transformation on the drawn digit
        if np.max(board) != 0 and drawing_started is True and drawing_stopped is True:
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
            #     LRmodel.predict(LR_input, dr.LR_params.item().get('weights'), dr.LR_params.item().get('base')))
            prediction2 = np.argmax(dr.model_conv.predict(test_x))

            numbers.append(prediction2)
            # predict1_text += str(prediction1)
            predict2_text += str(prediction2)
        # displaying the text on the screen
        # cv2.putText(frame, predict1_text, (5, 380), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, predict2_text, (350, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # cv2.putText(frame, predict3_text, (350, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, width, (5, 460), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # predict3_text += str(ans)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        # clearing the board
        elif k == ord('c'):
            board.fill(0)
            center_points.clear()

        elif k == ord('s'):
            digits.append(prediction2)
            str1 = ''.join(str(e) for e in digits)
            predict3_text += str1
            counter_eq = 1

            # if counter2 is 1:
            #     printeq(frame, predict3_text)
            #     counter2 = 1

            # cv2.putText(frame, predict3_text, (350, 300), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            if counter is 1:
                counter = 0
                printeq(frame, predict3_text)

            # printeq(frame, predict3_text)


        elif k == ord('e'):
            l = len(digits)
            if digits[0] is '*' or digits[0] is '/':
                cv2.putText(frame, "Error in equation", (350, 340), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)
                print("Error in equation")
                digits = []
            else:
                equation = ""
                for i in range(0, l):
                    equation += str(digits[i])
                digits = []
                print(equation)
                print(eval(equation))
                ans = eval(equation)
                predict4_text_ans = "" + predict4_text + str(ans)
                counter3 = 1
            counter_eq = 0

        if counter3 is 1:
            printans(frame, predict4_text_ans)
            counter3 = 1

        # if counter_akks is 1:
        #     printans(frame, predict3_text)
        #     counter_akks = 1

        if counter_eq is 1:
            printeq(frame, predict3_text)
            counter_eq = 1
            counter_akks = 1

        cv2.imshow('input', input)
        cv2.imshow('frame', frame)
        cv2.moveWindow("frame", 100, 20)

        cv2.imshow('board', board)

    cap.release()
    cv2.destroyAllWindows()