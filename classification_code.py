''' 
Place where all the 
classsification of the embeddings
occurs. 

Using KNN via face_recognition_api, with K=5 neighbours

'''

import sys 
import csv 
import os
import cv2
import numpy as np
import argparse
import face_recognition
from deepface import DeepFace
import pickle 

def parse_args():
    parser = argparse.ArgumentParser(description="Testing Options")
    parser.add_argument('--folder_name', type=str, help="Folder Name")
    parser.add_argument('--image_path', type=str, help="Single Image Path")
    parser.add_argument('--user_list_path', type=str, help="Input Attendence List for Registered Users!")
    parser.add_argument('--embedding_file_path', type=str, help="Known Embeddings File Path")
    parser.add_argument('--class_file_path', type=str,help="Known Class File Path")
    parser.add_argument('--K', type=str,help="K-Means Parameter")
    args = parser.parse_args()
    return args

def save_dict_to_pickle(file_path, data_dict):
    with open(file_path, 'wb') as pickle_file:
        pickle.dump(data_dict, pickle_file)

def load_dict_from_pickle(file_path):
    with open(file_path, 'rb') as pickle_file:
        loaded_dict = pickle.load(pickle_file)
    return loaded_dict

def get_top_k_indices(distance_array, k):
    index_distance_pairs = [(i, distance_array[i]) for i in range(len(distance_array))]
    sorted_pairs = sorted(index_distance_pairs, key=lambda x: x[1])  # Sort by distance
    top_k_indices = [pair[0] for pair in sorted_pairs[:k]]  # Extract top K indices
    return top_k_indices

def check_repeating_element(nums,k):
    frequency = {}
    for num in nums:
        if num in frequency:
            frequency[num] += 1
            if frequency[num] >=k:
                return num
        else:
            frequency[num] = 1
    return None

def generate_embedding(filename,modelname='ArcFace'):
    img = cv2.imread(filename)
    imgName=os.path.basename(filename)
    face_objs = DeepFace.extract_faces(img_path=filename,detector_backend ='retinaface',align=True)
    if len(face_objs)>0:
        face_ = 255*face_objs[0]['face']
        embedding_objs = DeepFace.represent(face_ ,model_name=modelname,enforce_detection=False)
        emb = embedding_objs[0]["embedding"]
        emb_arr = np.asarray(emb,dtype=np.float64)
        return emb_arr
    else:
        print('No Face Detected !')
        return None

def load_known_details(users_file_path,embedding_file_path,class_file_path):
    try:
        known_embs = load_dict_from_pickle(embedding_file_path)
        known_classes = load_dict_from_pickle(class_file_path)
        known_users_map = load_dict_from_pickle(users_file_path)
    except Exception as exp:
        print('Error in loading the important pickle files! : {}'.format(exp))
        sys.exit(1)
    return known_embs, known_classes,known_users_map

def test_single_embedding(test_emb,global_known_embedding,global_known_classes,K=5):
    # Extracting the embedding
    global_known_embedding = np.asarray(global_known_embedding,dtype=np.float32)
    global_known_classes = np.asarray(global_known_classes,dtype=np.int32)
    # Making into the embedding as array    
    test_emb = np.asarray(test_emb,dtype=np.float32)

    # Detect for this embedding
    matches = face_recognition.compare_faces(global_known_embedding,test_emb,tolerance=0.7)
    faceDis = face_recognition.face_distance(global_known_embedding,test_emb)

    # Picking up the top K matches
    top_k_indices=get_top_k_indices(faceDis,K)
    top_k_preds = [global_known_classes[i] for i in top_k_indices]
    val = check_repeating_element(top_k_preds,k=np.int32(K/2)+1)
    
    if val is not None:
      # Class votes 
      return val
    else:
      # No confidence
      return -1

def test_single_image(filename,user_list_path,embedding_file_path,class_file_path,K):
    # Gathering the known details .. 
    global_known_embedding,global_known_classes,ATTENDENCE = load_known_details(user_list_path,embedding_file_path,class_file_path)
    print('Testing single image : {}'.format(filename))
    canvas = cv2.imread(filename)
    h,w,c = canvas.shape
    # Name of the file (assuming its names as user.jpg )
    user_file_name = os.path.basename(filename)
    emb_ = generate_embedding(filename)
    if emb_ is not None:
        value = test_single_embedding(emb_, global_known_embedding,global_known_classes,K)
        if value==-1:
            print('UNKNOWN person detected !')
            canvas=cv2.putText(canvas,'UNKNOWN_USER ',(h//2,w//2),cv2.FONT_HERSHEY_COMPLEX,2,(0,255, 0),3)
            user = 'UNKNOWN'
        else:
            user = ATTENDENCE[value]
            print('Detected {} as {}!'.format(user_file_name,ATTENDENCE[value]))
            canvas=cv2.putText(canvas,user,(h//4,w//4),cv2.FONT_HERSHEY_COMPLEX,2,(255, 0, 0),3)
        return canvas,user
    return None,None

def test_folder_images(folder_name,user_list_path,embedding_file_path,class_file_path,K):
    # Call the function test_single_image multiple times
    for i,f in enumerate(os.listdir(folder_name)):
        try:
            data_path=os.path.join(folder_name,f)
            canvas = cv2.imread(data_path)
            h,w,c = canvas.shape
            _,user=test_single_image(data_path,user_list_path,embedding_file_path,class_file_path,K)
            if user is not None:
                print('For {} detected {} '.format(f,user))
            else:
                print('Error in User Identification,Check the sub-steps !')
        except Exception as e: 
            print('Error in Process,Skipping this user!:{}'.format(e))
            continue
    print('---Finished----')


if __name__ == '__main__':
    args = parse_args()
    test_folder_images(args.folder_name,args.user_list_path,args.embedding_file_path,args.class_file_path,np.int32(args.K))
    print('--------------------------------')
