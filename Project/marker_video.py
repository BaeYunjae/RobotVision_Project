import numpy as np
import cv2
import cv2.aruco as aruco
import time

# distortion matrix와 camara matrix

dist=np.array(([[-0.40650416 , 0.59103816, -0.00443272 , 0.00357844 ,-0.27203275]]))
newcameramtx=np.array([[189.076828   ,  0.    ,     361.20126638]
 ,[  0 ,2.01627296e+04 ,4.52759577e+02]
 ,[0, 0, 1]])
mtx=np.array([[398.12724231  , 0.      ,   304.35638757],
 [  0.       ,  345.38259888, 282.49861858],
 [  0.,           0.,           1.        ]])

font = cv2.FONT_HERSHEY_SIMPLEX

# 카메라로부터 비디오 읽기, 카메라 1개면 0
CAM_ID = (0)
cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
 
while(True):
    ret, frame = cap.read() # 비디오의 한 프레임씩 읽기 
    h1, w1 = frame.shape[:2] # 프레임의 행과 열을 각각 h1과 w1으로 저장
    # 카메라 화면 읽기
    # distortion 수정
    # camera matrix 개선, 결과를 자르는 데 사용할 수 있는 이미지 ROI 반환
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (h1, w1), 0, (h1, w1))
    dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx) # distortion 제거
    x, y, w1, h1 = roi
    dst1 = dst1[y:y + h1*4, x:x + w1*4]
    frame=dst1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    if ids is not None:
        # 각 marker의 pose를 추정하고 camera coefficient들과 다른 rotation vector와 translation vector 값 반환
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)

        (rvec-tvec).any() # numpy 값 배열 error 제거
        for i in range(rvec.shape[0]): # rotation vector의 행 갯수
            aruco.drawAxis(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03) 
            # drawAxis(image, cameraMatrix, distCoeffs, rvec, tvec, length)
            aruco.drawDetectedMarkers(frame, corners) 
            
        print('ID: ', ids)
        print('Corners: ', corners)
        
    detect = aruco.drawDetectedMarkers(frame, corners, ids)
    
    cv2.imshow('Detected Aruco Markers', detect)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() # 비디오 해제
cv2.destroyAllWindows()