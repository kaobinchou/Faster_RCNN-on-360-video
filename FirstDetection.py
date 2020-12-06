from imageai.Detection import ObjectDetection
import pandas as pd
import tensorflow
#print(tensorflow.__version__)
import os, sys

"""
n = len(sys.argv)
if n<2:
    quit()

inputfile = sys.argv[1]
"""
#outputfile = inputfile.split('.')[0].split('/')[1] + '_obg.jpg'
n=30
outputfile = "anotation.csv"
o = pd.DataFrame()
o_files = []
o_xmins = []
o_ymins = []
o_xmaxs = []
o_ymaxs = []
o_class = []

for i in range(n):
    inputfile = "frames/frame" + str(i) + ".jpg"

    print("start processing frame " + str(i) + "...")
    obj_list = {}
    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , inputfile),
    output_image_path=os.path.join(execution_path , "frame_objs/frame" + str(i) + ".jpg"))

    for eachObject in detections:
        if not eachObject["name"] in obj_list.keys():
            obj_list[eachObject["name"]] = 0
        o_files.append("../" + inputfile)
        o_xmins.append(eachObject["box_points"][0])
        o_ymins.append(eachObject["box_points"][1])
        o_xmaxs.append(eachObject["box_points"][2])
        o_ymaxs.append(eachObject["box_points"][3])
        o_class.append(eachObject["name"])
        obj_list[eachObject["name"]] += 1
        #print(eachObject["name"] , " : " , eachObject["box_points"] )

o['file'] = o_files
o['xmin'] = o_xmins
o['ymin'] = o_ymins
o['xmax'] = o_xmaxs
o['ymax'] = o_ymaxs
o['class'] = o_class
o.to_csv(os.path.join(execution_path, outputfile), header=None, index=None)
