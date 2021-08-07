#Generate dataset

import cv2  #pip install opencv-python
import csv
import glob

def one_time_only():
    header  =["label"]
    for i in range(0,784):
        header.append("pixel"+str(i))
    with open('dataset.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    for label in range(10):
        dirList = glob.glob("capture_images/" + str(label) + "/*.png")

        for img_path in dirList:
            im = cv2.imread(img_path)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)
            roi = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)

            data = []
            data.append(label)
            rows, cols = roi.shape

            ## Add pixel one by one into data array
            for i in range(rows):
                for j in range(cols):
                    k = roi[i, j]
                    if k > 100:
                        k = 1
                    else:
                        k = 0
                    data.append(k)
            with open('dataset.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(data)