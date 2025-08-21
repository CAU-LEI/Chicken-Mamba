import math
import os
import csv
import cv2
import numpy as np
import pandas as pd



def get_more_features(img_path, loca="all"):

    breast = {}
    gender = {}
    live = {}

    img = cv2.imread(img_path, 0)
    if loca == "left":
        img = img[:, :(img.shape[1] // 2)]
    elif loca == "right":
        img = img[:, (img.shape[1] // 2):]
    else:
        img = img
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [max_contour], 0, (255), thickness=cv2.FILLED)
    # Get area and convex area
    area = cv2.countNonZero(mask)
    hull = cv2.convexHull(max_contour)
    convex_area = cv2.contourArea(hull)
    # Get foreground perimeter and major/minor axis lengths
    cnt = max_contour
    perimeter = cv2.arcLength(cnt, True)
    # fitEllipse is used to fit an ellipse to approximate the given contour, (x,y) represents the center axis coordinates, (MA,ma) represents the major/minor axis lengths, angle represents the angle between the major axis and the horizontal axis
    (x0, y0), (MA, ma), angle = cv2.fitEllipse(hull)  # Using hull instead of cnt here is more accurate
    major_axis_length = max(MA, ma)
    minor_axis_length = min(MA, ma)
    focal_length = math.sqrt(major_axis_length * major_axis_length - minor_axis_length * major_axis_length)

    # Get bounding rectangle center coordinates and length/width
    x1, y1, width, height = cv2.boundingRect(cnt)
    height = max(height, width)
    width = min(height, width)
    # Get minimum bounding rectangle length, width and rotation angle
    min_rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(min_rect)
    box = np.intp(box)
    min_angle = (90 - min_rect[2] if min_rect[2] > 50 else min_rect[2])
    min_height = max(min_rect[1]);
    min_width = min(min_rect[1]);
    # min_width = np.linalg.norm(box[1] - box[2])
    # min_height = np.linalg.norm(box[0] - box[1])
    # Get minimum enclosing circle diameter
    (x2, y2), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x2), int(y2))
    radius = int(radius)

    data = {}
    data['id'] = os.path.basename(img_path)[:-4]
    data['ids'] = os.path.basename(img_path).split("_")[0]

    ### Calculate pixel points
    data['area'] = round(math.sqrt(area * flag * flag), 2)
    data['convex_area'] = round(math.sqrt(convex_area * flag * flag), 2)

    data['perimeter'] = round(perimeter * flag, 2)
    data['major_axis'] = round(major_axis_length * flag, 2)
    data["minor_axis"] = round(minor_axis_length * flag, 2)
    data['focal_length'] = round(focal_length * flag, 2)
    data['Eccentricity0'] = round(height / width, 2)  # Bounding rectangle aspect ratio
    data['Eccentricity1'] = round(major_axis_length / minor_axis_length, 2)  # Ellipticity
    data['Eccentricity2'] = round(focal_length / major_axis_length, 2)  # Ellipse eccentricity

    data['width'] = round(width * flag, 2)
    data['height'] = round(height * flag, 2)
    data['angle'] = round(min_angle, 2)
    data['min_width'] = round(min_width * flag, 2)
    data['min_height'] = round(min_height * flag, 2)

    data['r_area'] = round(math.sqrt(data['height'] * data['width']), 2)
    data['rm_area'] = round(math.sqrt(data['min_height'] * data['min_width']), 2)

    data['radius1'] = round(2 * radius * flag, 2)  # Minimum enclosing circle diameter
    data['radius2'] = round(math.sqrt(4 * data['area'] / math.pi), 2)  # Equivalent minimum enclosing circle diameter

    data['elem1'] = round(perimeter / math.sqrt(math.pi * area) / 2, 2)  # Circularity
    data['elem2'] = round(area / min_height / min_width, 2)  # Rectangularity
    data['elem3'] = round(area / convex_area, 2)  # Solidity
    data['elem4'] = round(area / radius / radius / math.pi, 2)  # Arc degree
    data['elem5'] = round(perimeter * perimeter / area, 2)  # Complexity
    data['elem6'] = round(min_height / min_width, 2)  # Aspect ratio

    return data


flag = 1

def get_file_feature1(input_path, output_path, locat, demo_path=None):
    data = get_more_features(demo_path, loca=locat)
    feature_colums = list(data.keys())
    df = pd.DataFrame(columns=feature_colums, index=[])
    imgs_list = os.listdir(input_path)
    folder = os.path.isdir(os.path.join(input_path, imgs_list[0]))
    for file in imgs_list:
        # print(file, folder)
        if folder:
            img_path = os.path.join(input_path, file)
            for img in os.listdir(img_path):
                res = get_more_features(os.path.join(img_path, img), loca=locat)
                n_df = pd.DataFrame.from_dict(res, orient='index', columns=['value'])
                n_df.index.name = 'key'
                n_df = n_df.T
                df = pd.concat([df, n_df], ignore_index=True)
        else:
            res = get_more_features(os.path.join(input_path, file), loca=locat)

            try:
                n_df = pd.DataFrame.from_dict(res, orient='index', columns=['value'])

            except:
                continue  
                
            n_df.index.name = 'key'
            n_df = n_df.T
            df = pd.concat([df, n_df], ignore_index=True)

    df.to_csv(output_path, index=False, header=True)


if __name__ == '__main__':

    input_path = r"img"
    save_path = r"./Feature.csv"
    demo_path = "img/4984_2.png"
    get_file_feature1(input_path, save_path, "all", demo_path)

