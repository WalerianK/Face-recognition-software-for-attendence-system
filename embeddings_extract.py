'''
Given each user folder 
with train/ and test/ sub-folders
we generate the embeddings 
for our work.

Face Detector : RetinaFace 
Face Embeddings : ArcFace 

'''

import os 
import cv2 
import sys 
import numpy as np 
import pickle
import argparse
from deepface import DeepFace

from  data_preparation import * 

def parse_args():
    parser = argparse.ArgumentParser(description="Embedding Extraction Options")
    parser.add_argument('--dataset_folder', type=str, help="Input Data Folder")
    parser.add_argument('--modelname', type=str, help="Embeddings Model Choice",default='ArcFace')
    parser.add_argument('--mode', type=str, help="Mode",default='train',required=True)
    args = parser.parse_args()
    return args


def get_root_folder_path(folder_path):
    # Get the absolute path of the given folder
    absolute_path = os.path.abspath(folder_path)
    # Get the root folder path using dirname repeatedly until no more parent folders are found
    while True:
        root_folder_path = os.path.dirname(absolute_path)
        if root_folder_path == absolute_path:
            break
        absolute_path = root_folder_path
    return root_folder_path


def generate_user_embeddings(parent_folder,modelname='ArcFace',mode='train'):
    train_folder = os.path.join(parent_folder,mode)
    user_name=parent_folder.split('/')[-1]
    print('[{}] Processing for {}'.format(mode.upper(),user_name))
    # Check if the parent folder exists and is non empty
    try:
        assert checkFolder(folder_path=train_folder),'Improper Parent Folder!'
    except Exception:
        sys.exit(0)
    # Iterate through the images and create the embeddings 
    features=[]
    for f in os.listdir(train_folder):
        try:
            image_path = os.path.join(train_folder,f)
            face_objs = DeepFace.extract_faces(img_path =image_path,detector_backend = 'retinaface',align=True)
            if len(face_objs)>0:
                face_ = 255*face_objs[0]['face']
                embedding_objs = DeepFace.represent(face_ ,model_name=modelname,enforce_detection=False)
                emb = embedding_objs[0]["embedding"]
                emb_arr = np.asarray(emb,dtype=np.float64)
                features.append(emb_arr)
        except Exception as e:
            print('Error creating embedding :{}'.format(e))
            continue
    # Store it as a pickle file
    with open(os.path.join(parent_folder,'{}_{}.pkl'.format(str(user_name),str(mode))),'wb') as p:
        pickle.dump(features,p)
    print('Total Feature Embeddings : {}'.format(len(features)))
 

# For all the users it will combined the embeddings and classification file 
def generate_global_embeddings(dataset_folder,modelname='ArcFace',mode='train',store=True):
    # Parse through the folders and read the pickle dump file 
    pooled_embeddings=[]
    pooled_classes=[]

    # Generating the attendence list 
    attendance= [ f for f in os.listdir(dataset_folder)]
    attendance.sort()
    
    # Pooling the embeddings.
    for f in os.listdir(dataset_folder):
        print('Processing {}'.format(f))
        parent_folder=os.path.join(dataset_folder,f)
        generate_user_embeddings(parent_folder,modelname=modelname,mode=mode)
        file_path = os.path.join(dataset_folder,f)+'/{}_{}.pkl'.format(f,mode)
        try:
            with open(file_path, "rb") as file:
                embeddings = pickle.load(file)
                class_number= attendance.index(f)
                for emb in embeddings:
                    pooled_embeddings.append(emb.tolist())
                    pooled_classes.append(np.int32(class_number))
        except Exception as exp:
            print('Error in loading embeddings and its corresponding class ! : {}'.format(exp))
            continue
    print('Stats :{} Embeddedings :{} Classes :{}'.format(mode.upper(),len(pooled_embeddings),len(pooled_classes)))

    # Finding out the user_list as well 
    user_list = os.listdir(dataset_folder)
    user_list.sort()
    user_dict={}
    for i,user_name in enumerate(user_list):
        user_dict[i]=str(user_name)

    if store:
        with open(os.path.join(dataset_folder,'{}_embeddings.pkl'.format(str(mode))),'wb') as p:
            pickle.dump(pooled_embeddings,p)
        with open(os.path.join(dataset_folder,'{}_classes.pkl'.format(str(mode))),'wb') as p:
            pickle.dump(pooled_classes,p)
        with open(os.path.join(dataset_folder,'{}_attendance.pkl'.format(str(mode))),'wb') as p:
            pickle.dump(user_dict,p)
    return pooled_embeddings,pooled_classes,
 
if __name__=='__main__':
    args = parse_args()
    _,_,_=generate_global_embeddings(args.dataset_folder,modelname=args.modelname,mode=args.mode,store=True)