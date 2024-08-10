import cv2
import sys

# Create a VideoCapture object and read from input file
video_name="sleeping"
cap = cv2.VideoCapture("./test_dataset/sleeping/video_test/sleeping.mp4")
# Check if camera opened successfully
if (cap.isOpened()== False):
    sys.exit("Error opening video file")
	
width_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(width_img)
height_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(height_img)  

# read text
result_track = []
with open ("./test_dataset/sleeping/result_track/sleeping.txt", "r") as file:
    file = file.read()
    file = file.split("\n")

for result in file:
    result = result.split(" ")
    result_track.append(result)
count = 0
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        count += 1
        if count < 10:
            s = "_00"
        else:
            if count < 100:
                s = "_0"
            else:
                if count <= 295:
                    s = "_"
                else:
                    sys.exit("Exit")
              
        for box in result_track:
            if box[0] == str(count):
                id = int(box[2])
                x1 = int(box[3])
                y1  = int(box[4])
                wbb= int(box[5])
                hbb = int(box[6])

                x_center = (x1+wbb/2)/width_img
                y_center = (y1+hbb/2)/height_img
                w_norm   = wbb/width_img
                h_norm   = hbb/height_img
                print(str(id) + " "+ str(x_center)+" "+str(y_center)+" "+str(w_norm)+" "+str(h_norm))
                with open ("./test_dataset/sleeping/" + video_name+"_label_image/labels/"+video_name+s+str(count)+".txt", "a+") as f:    
                    f.write(str(id) + " "+ str(x_center)+" "+str(y_center)+" "+str(w_norm)+" "+str(h_norm)+"\n")
        
        cv2.imwrite("./test_dataset/sleeping/" +video_name+"_label_image/images/" + video_name+s+str(count)+".jpg", frame)

    else:
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()