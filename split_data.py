import os
import shutil
def split():
    dir = 'VOC2012/ImageSets/Main/'
    train_file = f'{dir}train.txt'
    train_files = []
    with open(train_file, encoding='utf8') as f:
        for file_name in f:
            train_files.append(file_name.strip())
    if not os.path.exists('train/Annotations'):
        os.makedirs('train/Annotations')
    if not os.path.exists('train/Images'):
        os.makedirs('train/Images')
    source_annotations_dir = 'VOC2012/Annotations'
    source_images_dir = 'VOC2012/JPEGImages'
    for file in train_files:
        try:
            shutil.move(f'{source_annotations_dir}/{file}.xml', 'train/Annotations')
        except:
            pass
        try:
            shutil.move(f'{source_images_dir}/{file}.jpg', 'train/Images')
        except:
            pass


if __name__ == '__main__':
    split()
