import os
import json
import imagesize
import cv2
voc_labels = ('Wild', 'roedeer')
label_map = {k: v + 1 for v, k in enumerate(voc_labels)}
label_map['background'] = 0
rev_label_map = {v: k for k, v in label_map.items()}  # Inverse mapping
def parse_annotation(annotation_path):
    with open(annotation_path, "r") as f:
        annot = f.read().splitlines()
    objs = annot[1:]

    boxes = list()
    labels = list()
    difficulties = list()
    for object in objs:

        difficult = False
        data = object.split()
        label = data[-1]
        label = "Wild" if label == "wildpig" else label
        label = "roedeer" if label == "waterdeer" else label 
        if label not in label_map:
            continue

        bbox = data[:4]
        xmin = int(bbox[0]) - 1
        ymin = int(bbox[1]) - 1
        xmax = int(bbox[2]) - 1
        ymax = int(bbox[3]) - 1

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label_map[label])
        difficulties.append(difficult)

    return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

def create_data_lists(animal_path, test_path, output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.

    :param animal_path: path to the 'Images' folder
    :param test_path: path to the 'Images' folder
    :param output_folder: folder where the JSONs must be saved
    """
    animal_path = os.path.abspath(animal_path)
    animal_files = os.listdir(animal_path)
    #img_path = "Boxing_KNPS_image/KNPS_Captures"
    img_path = "Boxing_KNPS_image/Testing/Picture"
    train_images = list()
    train_objects = list()
    n_objects = 0

    # Training data
    for path in animal_files:
        txt_file = os.path.join(animal_path, path)
        # Find IDs of images in training data
        
        file_name = path.split(".txt")[0]
        
        objects = parse_annotation(txt_file)
        if len(objects['boxes']) == 0:
            continue
        n_objects += len(objects)
        train_objects.append(objects)
        path = os.path.join(img_path, file_name + '.JPG')
        if os.path.isfile(path):
            train_images.append(path)
        else:
            train_objects.pop()

    assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))
    test_path = os.path.abspath(test_path)
    animal_files = os.listdir(test_path)
    #img_path = "Boxing_KNPS_image/Testing/Picture"
    img_path = "Boxing_KNPS_image/KNPS_Captures"
    test_images = list()
    test_objects = list()
    n_objects = 0

    # Training data
    for path in animal_files:
        txt_file = os.path.join(test_path, path)
        # Find IDs of images in training data
        
        file_name = path.split(".txt")[0]
        
        objects = parse_annotation(txt_file)
        if len(objects['boxes']) == 0:
            continue
        n_objects += len(objects)
        test_objects.append(objects)
        path = os.path.join(img_path, file_name + '.JPG')
        if os.path.isfile(path):
            test_images.append(path)
        else:
            test_objects.pop()

    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)
    print('\nThere are %d testing images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))


def bbox_scaling(bbox, img_path):
    width, height = imagesize.get(img_path)
    bbox[0] = int(bbox[0]*width) 
    bbox[2] = int(bbox[2]*width) 
    bbox[1] = int(bbox[1]*height )
    bbox[3] = int(bbox[3]*height )

    return bbox

def create_data_lists_v2(output_folder):
    train_objects = list()
    train_images = list()
    with open("results_201011_md_v4.1.0.json", "r") as file:
        det_data = json.loads(file.read())["images"]
    objs = 0
    
    for index, det_file in enumerate(det_data):
        difficult = False
        difficulties = []
        img_path = det_file.get("file")
        dets = det_file.get("detections")
        boxes = []
        labels = []
        if not os.path.isfile(img_path):
            continue
        for detection in dets:
            bbox = detection.get("bbox")
            label = detection.get("category")
            if label not in voc_labels:
                continue
            boxes.append(bbox_scaling(bbox, img_path))
            labels.append(label_map[label])
            difficulties.append(difficult)
            objs += 1
        if len(boxes) == 0:
            continue

        objects = {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}
        train_objects.append(objects)
        train_images.append(img_path)
    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), objs, os.path.abspath(output_folder)))
    assert len(train_objects) == len(train_images)
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(label_map, j)  # save label map too






def edit_json():
    detections = None
    data = {"images": detections}
    with open("results_201011_md_v4.1.0.json", "r") as file:
        det_data = json.loads(file.read())
    detections = det_data.get("images")
    for index, det in enumerate(detections):
        img_name = det.get("file").split("/")[-1]
        new_path = os.path.join("train", img_name)
        detections[index]["file"] = new_path
    data.update({"images": detections})
    with open("results_201011_md_v4.1.0.json", "w") as file:
        json.dump(data, file)

create_data_lists("Boxing_KNPS_image/Testing/Labels", "Boxing_KNPS_image/Labels/", "./")
#create_data_lists_v2("./")
#edit_json()