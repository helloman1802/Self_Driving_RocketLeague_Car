import time
import cv2
import mss
import numpy as np
import pyautogui as pygui
import pyscreenshot as ImageGrab
import pandas as pd



def fps1():
    with mss.mss() as sct:
        # Part of the screen to capture
        monitor = {'top': 30, 'left': 0, 'width': 800, 'height': 600}
       
        
        while 'Screen capturing':
            last_time = time.time()

            # Get raw pixels from the screen, save it to a Numpy array
            screen = sct.grab(monitor)
            screen = np.array(screen)
            color = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
            #color = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)  
            #gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
           
            cv2.imshow('OpenCV/Numpy normal', color)
           
            print('fps: {0}'.format(1 / (time.time()-last_time)))
            print('Color: {}'.format(np.shape(color)))
            #print('Gray: {}'.format(np.shape(gray)))


            # Press "q" to quit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        

fps1()
