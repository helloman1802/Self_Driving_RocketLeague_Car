from getkeys import key_check
import numpy as np
import os
import mss # For grabing a region of the screen to train off of.
import time
import cv2
import tempfile, shutil


def keys_to_output(keys):
    '''
    Convert keys to a ...multi-hot... array

    [A,W,D] boolean values.
    '''
    output = [0,0,0]
    
    if 'a' in keys:
        output[0] = 1
    elif 'd' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output


file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []

def main():

    # Count Down
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
    
    
    with mss.mss() as sct:
        # Part of the screen to capture
        monitor = {'top': 30, 'left': 0, 'width': 800, 'height': 600}    
        while(True):
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
            # Get keys that are currently down.
            keys = key_check()
            # Converts the keys pressed into array
            output = keys_to_output(keys)
            # Append to the training list
            training_data.append([image_np, output])
            
            # Print current FPS
            print('fps: {0}'.format(1 / (time.time()-last_time)))

            # Save training data every 500 steps
            if len(training_data) % 500 == 0:
                print(len(training_data))
                np.save(file_name, training_data)

            


main()