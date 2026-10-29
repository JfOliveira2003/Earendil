import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

#if object.xml-> <object_tag> -> <name> == "pothole": mv object.xml <destination_folder>
XML_SRC_PATH = "../dataset/Annotations"
YOLO_DEST_PATH = "../dataset/Annotations_Pothole/"
JPEG_SRC_PATH = "../dataset/JPEGImages/"
JPEG_DEST_PATH = "../dataset/JPEGImages_Pothole/"

def convert_bbox(height, width, xmin, ymin, xmax, ymax):
    xcenter = (xmin + xmax) / 2 / width
    ycenter = (ymin + ymax) / 2 / height
    width = (xmax - xmin) / width
    height = (ymax - ymin) / height
    return xcenter, ycenter, width, height

def process_XML_file(filename):
    tree = ET.parse(filename)

    root = tree.getroot()
    #print(f"Root tag: {root.tag}")

    lines = []
    for object in root.findall("object"):
        object_kind = object.find('name').text
        if(object_kind == "Pothole"):
            #shutil.copy(filename, XML_DEST_PATH)
            jpeg_filename = root.find('filename').text
            print(jpeg_filename)
            shutil.copy(JPEG_SRC_PATH + jpeg_filename, JPEG_DEST_PATH)

            #extracting image dimensions
            xmin = int(object.find('bndbox').find('xmin').text)
            ymin = int(object.find('bndbox').find('ymin').text)
            xmax = int(object.find('bndbox').find('xmax').text)
            ymax = int(object.find('bndbox').find('ymax').text)

            height = int(root.find('size').find('height').text)
            width = int(root.find('size').find('width').text)

            xcenter, ycenter, width, height = convert_bbox(height, width, xmin, ymin, xmax, ymax)
            lines.append(f"{0} {xcenter:.6f} {ycenter:.6f} {width:.6f} {height:.6f}")

            #creating YOLO format file
            dest_file = YOLO_DEST_PATH + jpeg_filename.replace(".jpg", ".txt")
            with open(dest_file, "w") as f:
                f.write("\n".join(lines))


            
        
            

folder_path = Path(XML_SRC_PATH)

for file in folder_path.iterdir():
    if file.is_file():
        process_XML_file(file)

