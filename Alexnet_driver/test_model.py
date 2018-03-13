from getkeys import key_check
import numpy as np
import os
import mss # For grabing a region of the screen to train off of.
import time
import cv2
import pyautogui as pygui
from alexnet import alexnet


# Sizes of the image data
WIDTH = 80
HEIGHT = 60
# Learning rate
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'rl-car-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2',EPOCHS)

def straight():
    pygui.keyDown('w')
    pygui.keyUp('a')
    pygui.keyUp('d')

def left():
    pygui.keyDown('a')
    pygui.keyDown('w')
    pygui.keyUp('d')
    

def right():
    pygui.keyDown('d')
    pygui.keyDown('w')
    pygui.keyUp('a')

model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():

    # Count Down
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    
    
    with mss.mss() as sct:
        # Part of the screen to capture
        monitor = {'top': 30, 'left': 0, 'width': 800, 'height': 600}    
        paused = False
        while(True):
            #if not paused:
            # Using to track FPS
            last_time = time.time()

            # Grabs the region of the monitor
            image = sct.grab(monitor)
            # Converts the image into a numpy array
            image_np = np.array(image)
            # Changes the color to RGB and resizes the numpy array
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            # resize to something a bit more acceptable for a CNN
            image_np = cv2.resize(image_np, (80,60))
            
            
            # Print current FPS
            #print('fps: {0}'.format(1 / (time.time()-last_time)))
            prediction = model.predict([image_np.reshape(WIDTH, HEIGHT,1)])
            moves = np.around(prediction)
            #print(moves, prediction)
            moves = moves.tolist()
            moves = moves[0]
            if moves == [1,0,0]:
                left()
                print('left')

            elif moves == [0,1,0]:
                straight()
                print('straight')

            elif moves == [0,0,1]:
                right()
                print('right')

            else:
                print('failing')

            """
            keys = key_check()

            if 't' in keys:
                if paused:
                    paused = False
                    time.sleep(1)
                else:
                    paused = True
                    pygui.keyUp('a')
                    pygui.keyUp('w')
                    pygui.keyUp('d')
                    time.sleep(1)
            """

            


main()