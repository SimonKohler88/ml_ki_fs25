#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import os
import shutil

#  converting the images in archive to numpy format, scaling down 224x224 imgs to 179x179.
# --> there are 2 types of pic: 224x224 and 179x179. use latter for less data

if __name__ == '__main__':
    path = os.path.dirname(os.path.abspath(__file__))
    archive = os.path.join(path, 'archive')

    heights = []
    widths = []
    for abspath, folders, files in os.walk(archive):
        if len(files) == 0:
            continue

        for each in files:
            name, ext = os.path.splitext(each)
            # print(name, ext)

            if ext.lower() in ['.jpg', '.jpeg']:
                src_file = os.path.join(abspath, each)
                im_temp = Image.open(src_file)

                # resize bigger 224x224 pictures down to 179x179
                if im_temp.height == 224:
                    im_temp.thumbnail((179, 179), Image.Resampling.LANCZOS)

                if im_temp.height not in heights:
                    heights.append(im_temp.height)
                if im_temp.width not in widths:
                    widths.append(im_temp.width)

                img = np.asarray(im_temp)
                new_file_name = os.path.join(abspath, f'{name}.npy')
                np.save(new_file_name, img)

    print('widths: ', widths)
    print('heights: ', heights)
    # pic = 'Earth_relief_120x256.bmp'
    # im_temp = Image.open(pic)
    #
    # img = np.asarray(im_temp)
    # np.save('test3.npy', a)  # .npy extension is added if not given
    # d = np.load('test3.npy')
