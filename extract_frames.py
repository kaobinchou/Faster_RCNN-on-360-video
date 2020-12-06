import cv2
vidcap = cv2.VideoCapture('CS598_Chickens360_1.mp4')
success,image = vidcap.read()
count = 0
while success:
    cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
    success,image = vidcap.read()
    if not success:
        print('Frame number ' + count + ' has failed')
    count += 1

print('Total frames: ' + count)
