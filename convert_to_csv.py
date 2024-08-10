import pandas as pd
#convert to xmin, xmax, ymin, ymax

with open ("./test_dataset/sleeping/data/grt.txt", "r") as f:
    f = f.read().split("\n")

df = pd.DataFrame(columns=["frame", "personID", "xmin", "xmax", "ymin", "ymax"])
for boxes in f:
    if boxes != '':
        boxes = boxes.split(" ") 
        boxes = list(map(int, boxes)) # frame, personID, x1, y1, w, h    
        xmax = boxes[2] + boxes[4]
        ymax = boxes[3] + boxes[5]
        boxes[5] = ymax
        boxes[4] = boxes[3]
        boxes[3] = xmax
        
        df.loc[len(df)] = boxes
df.to_csv("./test_dataset/sleeping/data/grt.csv", index=False)

            


