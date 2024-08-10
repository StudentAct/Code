
import os
import math

file_path = './test_dataset/sleeping/data/grt.txt'
txt_path = './test_dataset/sleeping/sleeping_label_image/labels/'

i = 0
merged_text = ""

def round_up(n, decimals = 0):
    multiplier = 10 ** decimals
    return math.ceil(n*multiplier) / multiplier

for txt_file in os.listdir(txt_path):
    if (txt_file != 'classes.txt'):
        file = open(f'{txt_path}{txt_file}')
        file = file.read()
        i += 1
        file = file.split("\n")
      
        for lns in file:
            if lns != '':
                lns = lns.split(" ")  
                id = lns[0]
                w = float(lns[3]) * 1920
                h = float(lns[4]) * 1080
                xc = float(lns[1])
                x1 = xc * 1920 - w/2
                yc = float(lns[2])
                y1 = yc * 1080 - h/2
                if(i == 167): 
                    i += 41
              
                merged_text += str(i) + " " + str(id) + " " + str(int(x1)) + " " + str(int(y1)) + " " + str(int(round_up(w))) + " " + str(int(round_up(h))) + "\n"

    with open(file_path, 'w') as file:
            file.write(merged_text) 
    
       

       
        