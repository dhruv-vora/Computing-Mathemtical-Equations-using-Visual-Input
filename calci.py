import cv2
import numpy as np
import quadra_eq as qd
import linear_eq as lq
import trigno_eq_akks as tr
import camera_pred as cal
cap = cv2.VideoCapture(0)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

lowergreen = np.array([50, 100, 50])
uppergreen = np.array([90, 255, 255])

drawing_started = False

def draw_rectangle_at_coordinates(x1, y1, x2, y2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 255), 2)


def get_roi(frame, x1, y1, x2, y2):
    roi = frame[y1: y2, x1: x2, :]
    return roi


def get_hsv_roi(roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    return hsv_roi


def get_roi_range(hsv_roi):
    roi_range = cv2.inRange(hsv_roi, lowergreen, uppergreen)
    return roi_range

def get_contours_and_heirarchy(roi):
    contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

while cap.isOpened():
    ret, frame = cap.read()
    # flipping the frame
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (int(width), int(height)))
    draw_rectangle_at_coordinates(200, 100, 600, 300)
    draw_rectangle_at_coordinates(700, 100, 1100, 300)
    draw_rectangle_at_coordinates(200, 400, 600, 600)
    draw_rectangle_at_coordinates(700, 400, 1100, 600)
    roi_1 = get_roi(frame, 200, 100, 600, 300)
    roi_2 = get_roi(frame, 700, 100, 1100, 300)
    roi_3 = get_roi(frame, 200, 400, 600, 600)
    roi_4 = get_roi(frame, 700, 400, 1100, 600)
    hsv_roi_1 = get_hsv_roi(roi_1)
    hsv_roi_2 = get_hsv_roi(roi_2)
    hsv_roi_3 = get_hsv_roi(roi_3)
    hsv_roi_4 = get_hsv_roi(roi_4)
    roi_range_1 = get_roi_range(hsv_roi_1)
    roi_range_2 = get_roi_range(hsv_roi_2)
    roi_range_3 = get_roi_range(hsv_roi_3)
    roi_range_4 = get_roi_range(hsv_roi_4)
    contours_1, hierarchy_1 = get_contours_and_heirarchy(roi_range_1.copy())
    contours_2, hierarchy_2 = get_contours_and_heirarchy(roi_range_2.copy())
    contours_3, hierarchy_3 = get_contours_and_heirarchy(roi_range_3.copy())
    contours_4, hierarchy_4 = get_contours_and_heirarchy(roi_range_4.copy())

    cv2.putText(frame, "Calculator", (310, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Quadratic Equation", (250, 500), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Linear Equation", (780, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Trignometric Equation", (720, 500), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    if len(contours_1) > 0:
        drawing_started = True
        cal.calci_main()

    if len(contours_2) > 0:
        drawing_started = True
        lq.lineq()

    if len(contours_3) > 0:
        drawing_started = True
        qd.quadr()

    if len(contours_4) > 0:
        drawing_started = True
        tr.trigno()
    # Display the resulting frame

    cv2.imshow('frame', frame)
    cv2.moveWindow("frame", 100,20)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
