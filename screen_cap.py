#SCREEN CAPTURE

import pyscreenshot as ImageGrab  # pip install pyscreenshot
import time                       # pip install pillow
def one_time():
    images_folder="capture_images/9/"
    for i in range (0,40):
        time.sleep(8)
        im=ImageGrab.grab(bbox=(60,170,400,550))
        print("saved...",i)
        im.save(images_folder+str(i)+'.png')
        print("clear the screen and redraw now....")
