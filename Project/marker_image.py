import cv2         
import numpy as np  
import cv2.aruco as aruco # aruco library 설치 후 import 

# marker 이미지 불러오기
image = cv2.imread('markers.jpg')
  
# cv2.cvtColor를 이용해 image를 grayscale로 변환
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
# Otsu thresholding
ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + 
                                            cv2.THRESH_OTSU)     

# 이진화(thresholding)된 이미지를 화면에 출력 
cv2.imshow('Otsu Threshold', thresh)

# 이진화된 이미지에서 contour들을 찾아 원본 이미지에 중첩
contour_img = np.copy(thresh)
contours, hierachy = cv2.findContours(contour_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
result = cv2.drawContours(image,contours, -1, (255,0,255), 1)

arucoDict = aruco.Dictionary_get(aruco.DICT_6X6_250)
arucoParams = aruco.DetectorParameters_create()
(corners, ids, rejected) = aruco.detectMarkers(contour_img, arucoDict, parameters=arucoParams)

# 적어도 하나의 ArUco marker가 인식되는 것을 확인
if len(corners) > 0:
    ids = ids.flatten() # ArUco ID들이 담긴 리스트를 하나의 행 또는 열로 변환

    # ArUCo corners 검출 반복
    for (markerCorner, markerID) in zip(corners, ids):
        # marker corners 추출
        # 항상 왼쪽 상단(top-left),오른쪽 상단(top-right),오른쪽 하단(bottom-right),왼쪽 하단(bottom-left) 순서로 반환
        corners = markerCorner.reshape(4, 2)
        (topLeft, topRight, bottomRight, bottomLeft) = corners

        # 각 corner점들의 (x,y) 좌표값들을 각각 정수들로 변환
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))

        # ArUco 인식에 대해 bounding box를 그린다.
        cv2.line(result, topLeft, topRight, (0, 255, 0), 2)
        cv2.line(result, topRight, bottomRight, (0, 255, 0), 2)
        cv2.line(result, bottomRight, bottomLeft, (0, 255, 0), 2)
        cv2.line(result, bottomLeft, topLeft, (0, 255, 0), 2)

        # ArUco marker의 중심 (x, y)를 계산하고 중심점을 그린다.
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(result, (cX, cY), 4, (0, 0, 255), -1)

        # marker 이미지에 ArUco marker ID 나타내기
        cv2.putText(result, str(markerID),
            (bottomLeft[0], bottomRight[1] - 20), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2)
        
        # marker 이미지에 marker의 corner 표시
        cv2.putText(result, '1', topLeft, cv2.FONT_ITALIC,
            1, (255, 0, 0), 2)
        cv2.putText(result, '2', topRight, cv2.FONT_ITALIC,
            1, (255, 0, 0), 2)
        cv2.putText(result, '3', bottomRight, cv2.FONT_ITALIC,
            1, (255, 0, 0), 2)
        cv2.putText(result, '4', bottomLeft, cv2.FONT_ITALIC,
            1, (255, 0, 0), 2)
        
        # markerID와 corner 좌표들 출력
        print('*ArUco marker ID: {}'.format(markerID))
        print('corner 1: ', topLeft)
        print('corner 2: ', topRight)
        print('corner 3: ', bottomRight)
        print('corner 4: ', bottomLeft)

cv2.imshow('result', result)
       
# ESC로 종료       
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()  