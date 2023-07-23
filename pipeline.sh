#!/bin/bash

python data_preparation.py --parent_folder '/home/niharika.v/PRASANTH/images_tiny' --target_folder '/home/niharika.v/PRASANTH/dataset'
python embeddings_extract.py --dataset_folder '/home/niharika.v/PRASANTH/dataset' --modelname 'ArcFace' --mode 'train'
python classification_code.py  --K '5' --folder_name '/home/niharika.v/PRASANTH/mixed_test_folder' --user_list_path '/home/niharika.v/PRASANTH/dataset/attendance.pkl' --embedding_file_path '/home/niharika.v/PRASANTH/dataset/train_embeddings.pkl' --class_file_path '/home/niharika.v/PRASANTH/dataset/train_classes.pkl'             
