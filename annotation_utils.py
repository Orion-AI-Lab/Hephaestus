import json
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import albumentations as A
import os
from albumentations import BboxParams
import pandas as pd
from utilities import data_utilities
from collections import Counter
import pprint
np.random.seed(999)
random.seed(999)


def read_annotation(annotation_path):
    reader = open(annotation_path)
    annotation = json.load(reader)
    return annotation

def extract_label(annotation_path):
    annotation = read_annotation(annotation_path)
    if annotation['label']=='Non_Deformation':
        label = 0
    elif annotation['label']=='Deformation':
        label = 1
    else:
        label = 2
    return label

def get_insar_path(annotation_path,root_path='/home/nbountos/hdd1/nbountos-ssh/LiCSAR/LiCSAR-web-tools/'):
    annotation = read_annotation(annotation_path)
    frameID = annotation['frameID']
    primaryDate = annotation['primary_date']
    secondaryDate = annotation['secondary_date']
    primary_secondary = primaryDate + '_' + secondaryDate
    img_path = root_path + frameID + '/interferograms/' + primary_secondary + '/' + primary_secondary + '.geo.diff.png'
    return img_path

def get_segmentation(annotation_path='1.json', verbose=True):
    class_dir = {'Mogi':1, 'Dyke':2, 'Sill': 3, 'Spheroid':4, 'Earthquake':5, 'Unidentified':6}
    annotation = read_annotation(annotation_path)
    img_path = get_insar_path(annotation_path)
    segmentation = annotation['segmentation_mask']
    insar = cv.imread(img_path,0)
    if verbose:
        plt.imshow(insar)
        plt.show()
    masks = []
    if len(segmentation) == 0:
        return []
    if not any(isinstance(el, list) for el in segmentation):
        segmentation = [segmentation]
    for idx,seg in enumerate(segmentation):
        i = 0
        points = []
        mask = np.zeros(insar.shape)

        while i +1 < len(seg):
            x = int(seg[i])
            y = int(seg[i+1])
            points.append([x,y])
            i+=2
        print('File : ',annotation_path)
        cv.fillPoly(mask, [np.asarray(points)], class_dir[annotation['activity_type'][idx]])#255)
        if verbose:
            plt.imshow(mask)
            plt.show()
        masks.append(mask)
        print('Number of mask: ',len(masks))
    return masks

def mask_boundaries(annotation_path,verbose=True,index=0):
    '''

    :param annotation_path: path of annotation file
    :return: segmentation mask boundaries (top_left_point,bottom_right_point)
    '''
    class_dir = {'Mogi':1, 'Dyke':2, 'Sill': 3, 'Spheroid':4, 'Earthquake':5, 'Unidentified':6}

    annotation = read_annotation(annotation_path)
    if type(annotation['segmentation_mask']) is not list:
        print(annotation_path)
    mask = get_segmentation(annotation_path,verbose=False)[index]
    print(mask.shape)
    row,col = (mask==class_dir[annotation['activity_type'][index]]).nonzero()
    if verbose:
        rect = cv.rectangle(mask, pt1=(col.min(),row.min()), pt2=(col.max(),row.max()), color=255, thickness=2)
        plt.imshow(rect)
        plt.show()
    return (row.min(),col.min()), (row.max(),col.max())


def crop_around_object(annotation_path,verbose=True,output_size=224,index=0):
    '''

    :param annotation_path: annotation file path
    :param verbose: Option to plot and save the cropped image and mask.
    :param output_size: Desired size of cropped image
    :param index: Index of ground deformation in the InSAR. Useful for cases where the InSAR contains multiple ground deformation patterns.
    :return: Randomly cropped image to output_size along with the respective mask. The cropped image is guaranteed to contain the ground deformation pattern.
    '''
    #Get all masks
    masks = get_segmentation(annotation_path,False)


    (row_min,col_min),(row_max,col_max) = mask_boundaries(annotation_path,False,index=index)
    mask = get_segmentation(annotation_path,False)[index]
    object_width = col_max - col_min
    object_height = row_max - row_min

    random_right = np.random.randint(col_max+max(output_size-col_min,0),min(col_max+abs(output_size-object_width),mask.shape[1]))
    left = random_right - output_size

    random_down = np.random.randint(row_max+max(output_size-row_min,0),min(row_max+abs(output_size-object_height),mask.shape[0]))
    up = random_down - output_size

    #Unite Other Deformation Masks
    if len(masks)>0:
        for k in range(len(masks)):
            if k!=index:
                mask = mask + masks[k]
    mask = mask[up:random_down,left:random_right]
    image_path = get_insar_path(annotation_path)
    image = cv.imread(image_path)
    if verbose:

        insar_path = get_insar_path(annotation_path)
        insar = cv.imread(insar_path)
        insar = cv.cvtColor(insar,cv.COLOR_BGR2RGB)
        print(insar[up:random_down,left:random_right].shape)
        print(mask.shape)
        plt.imshow(insar[up:random_down,left:random_right,:])
        plt.axis('off')
        plt.show()

        cmap = plt.get_cmap('tab10', 7)
        plt.imshow(mask,cmap=cmap,vmax=6.5,vmin=-0.5)
        plt.axis('off')
        plt.show()

    return image[up:random_down,left:random_right,:], mask


def save_crops(annotation_folder='annotations/',save_path = '/home/nbountos/hdd1/nbountos-ssh/InSARDataset/InSARLabelled/',mask_path='/home/nbountos/hdd1/nbountos-ssh/InSARDataset/masks/'):
    '''

    :param annotation_folder: folder of annotation jsons
    :param save_path: folder path for generated images
    :param mask_path: folder path for generated masks
    :return: Label vector [ Deformation/Non Deformation ( 0 for Non Deformation, 1,2,3,4,5 for Mogi, Dyke, Sill, Spheroid, Earthquake), Phase (0 -> Rest, 1-> Unrest, 2-> Rebound), Intensity Level (0->Low, 1-> Medium, 2->High)]
    '''
    annotations = os.listdir(annotation_folder)
    c = 0
    label_path = mask_path[:-6]+'cls_labels/'
    multiple_ctr = 0
    class_dir = {'Mogi':1, 'Dyke':2, 'Sill': 3, 'Spheroid':4, 'Earthquake':5, 'Low':0,'Medium':1,'High':2,'None':0,'Rest':0,'Unrest':1,'Rebound':2,'Unidentified':6}
    for file in annotations:
        label_json = {}
        annotation = read_annotation(annotation_folder+file)

        if 'Non_Deformation' in annotation['label']:
                image_path = get_insar_path(annotation_folder+file)
                image = cv.imread(image_path)
                tiles = image_tiling(image)
                for idx,tile in enumerate(tiles):
                    if image is None:
                        print(image_path)
                        print(file)
                        continue
                    #image = data_utilities.random_crop(image)
                    cv.imwrite(save_path+'0/'+file[:-5]+'_'+str(idx)+'.png',tile)
                    label_json['Deformation'] = [0]
                    label_json['Intensity'] = 0
                    label_json['Phase'] = class_dir[annotation['phase']]
                    label_json['frameID'] = annotation['frameID']
                    label_json['primary_date'] = annotation['primary_date']
                    label_json['secondary_date'] = annotation['secondary_date']
                    json_writer = open(label_path+file[:-5]+'_'+str(idx)+'.json','w')
                    json.dump(label_json,json_writer)
        elif int(annotation['is_crowd'])==0:
            if 'Non_Deformation' not in annotation['label']:
                image, mask = crop_around_object(annotation_path=annotation_folder+file,verbose=False)
                folder = str(class_dir[annotation['activity_type'][0]])
                cv.imwrite(save_path+folder+'/'+file[:-5]+'.png',image)
                cv.imwrite(mask_path+folder+'/'+file[:-5]+'.png',mask)
                label_json['Deformation'] = [class_dir[annotation['activity_type'][0]]]
                label_json['Intensity'] = class_dir[annotation['intensity_level']]
                label_json['Phase'] = class_dir[annotation['phase']]
                label_json['frameID'] = annotation['frameID']
                label_json['primary_date'] = annotation['primary_date']
                label_json['secondary_date'] = annotation['secondary_date']
                json_writer = open(label_path + file, 'w')
                json.dump(label_json, json_writer)
        elif int(annotation['is_crowd'])>0:
            for deformation in range(len(annotation['segmentation_mask'])):
                image, mask = crop_around_object(annotation_path=annotation_folder + file, verbose=False,index=deformation)
                print('Deformation index:',deformation)
                print('Seg length: ',len(annotation['segmentation_mask']))
                print('Annotation length: ',len(annotation['activity_type']))
                print('File: ',file)
                #if len(annotation['activity_type'])==1:
                #    folder = str(class_dir[annotation['activity_type'][0]])
                folder = str(class_dir[annotation['activity_type'][deformation]])
                cv.imwrite(save_path + folder + '/' + file[:-5] + '_'+ str(deformation) + '.png', image)
                cv.imwrite(mask_path + folder + '/' + file[:-5] + '_'+ str(deformation) + '.png', mask)
                label_json['Deformation'] = list(np.unique(mask))#class_dir[annotation['activity_type'][deformation]]
                label_json['Deformation'] = [int(x) for x in label_json['Deformation']]
                label_json['Deformation'].remove(0)
                label_json['Intensity'] = class_dir[annotation['intensity_level']]
                label_json['Phase'] = class_dir[annotation['phase']]
                label_json['frameID'] = annotation['frameID']
                label_json['primary_date'] = annotation['primary_date']
                label_json['secondary_date'] = annotation['secondary_date']
                json_writer = open(label_path + file[:-5] + '_'+ str(deformation) + '.json', 'w')
                json.dump(label_json, json_writer)
                multiple_ctr +=1
    print('Multiple deformation tiles: ',multiple_ctr)


def image_tiling(image,tile_size=224):
    if image is None:
        return []
    max_rows = image.shape[0]//tile_size
    max_cols = image.shape[1]//tile_size
    tiles = []
    for i in range(max_rows):
        starting_row = i * 224
        for j in range(max_cols):
            starting_col = j * 224
            img = image[starting_row:starting_row+224,starting_col:starting_col+224]
            tiles.append(img)

    return tiles