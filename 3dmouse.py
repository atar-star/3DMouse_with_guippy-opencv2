# -*- coding: utf-8 -*-
import guippy,cv2
import numpy as np

gp = guippy.Guippy()
mousex_old, mousey_old = 0, 0

kernel = np.ones((5,5),np.uint8) # 膨張化のためのカーネル定義

cap = cv2.VideoCapture(0) # カメラ取得

# カメラ完全起動待ち
ret = False
while not ret:
    ret, im = cap.read()
    
# OpenCVでの画像表示ウィンドウ初期化
cv2.namedWindow("Camera View", cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow("Camera View",im.shape[1]*3/5 * 2,im.shape[0]*3/5)

while True:    
    ret, im = cap.read() # カメラ画像取得
    im = cv2.resize(im,(im.shape[1]*3/5,im.shape[0]*3/5))
    # im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    
    # 領域抽出
    im_m = cv2.inRange(im, np.array([0,70,0]), np.array([60,255,255]),)
    # im_m = cv2.inRange(im_hsv, np.array([50,100,100]), np.array([70,255,255]),)
    im_m = cv2.medianBlur(im_m,7) # 平滑化
    im_m = cv2.morphologyEx(im_m, cv2.MORPH_OPEN, kernel) # 膨張化
    im_c = cv2.bitwise_and(im,im, mask=im_m) # マスク画像から指定した色の領域を抽出
    
    gray = cv2.cvtColor(im_c, cv2.COLOR_BGR2GRAY)
    contours,hierarchy = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # 最大領域を抽出
        areas = list(map(cv2.contourArea, contours))
        ind_max_area = np.argmax(areas)
        M = cv2.moments(contours[ind_max_area])

        cx1, cy1 = int(M['m10']/M['m00']), int(M['m01']/M['m00']) # 領域の中心点

        # 領域の表示
        #cv2.drawContours(im,contours,-1,(0,255,0),-1)
        cv2.drawContours(im,[contours[ind_max_area]],0,(0,255,0),-1)
        cv2.circle(im, (cx1,cy1), 5, 255, -1) # 中心点 
            
        # マウス点の算出
        mousex = 65535 * (1.1 * cx1 / im.shape[1] - 0.05)
        mousey = 65535 * (1.1 * cy1 / im.shape[0] - 0.05)

        mousex = 65535 - int(np.clip(mousex, 0, 65535)) # 自撮り型カメラ画像は左右反転するので、ここで反転
        mousey = int(np.clip(mousey, 0, 65535))

        # 手振れ軽減
        mousex = int(0.8*mousex + 0.2*mousex_old)
        mousey = int(0.8*mousey + 0.2*mousey_old)
            
        print(mousex, mousey, cx1, cy1)
        gp.jump(mousex, mousey)

        mousex_old, mousey_old = mousex, mousey
        
    # カメラ画像表示
    cv2.imshow("Camera View", im)
    cv2.imshow("Tracking Area", im_c)
    
    if cv2.waitKey(10) > 0: # 'q'で終了
        cap.release()
        cv2.destroyAllWindows()
        break