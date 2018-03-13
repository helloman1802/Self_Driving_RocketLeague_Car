import keyboard #Using module keyboard
from time import sleep

keyList = ["\b"]
for char in "abcdefghijklmnopqrstuvwxyz 123456789":
    keyList.append(char)


def key_check():
    keys = []
    for key in keyList:
        if keyboard.is_pressed(key) == True:
            keys.append(key)
    return keys

            
    
   