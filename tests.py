import os
import json

voc_labels = ('waterdeer', 'wildpig', 'roedeer', 'Wild')
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
    img_path = "Boxing_KNPS_image/KNPS_Captures"

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
        train_images.append(os.path.join(img_path, file_name + '.JPG'))

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
    img_path = "Boxing_KNPS_image/Testing/Picture"

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
        test_images.append(os.path.join(img_path, file_name + '.JPG'))

    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
        json.dump(test_objects, j)
    print('\nThere are %d testing images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_objects, os.path.abspath(output_folder)))

create_data_lists("Boxing_KNPS_image/Labels/", "Boxing_KNPS_image/Testing/Labels", "./")
