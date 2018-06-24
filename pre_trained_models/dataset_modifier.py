import cv2
import os
import random

def operation( image, per_cropped=0.2, hor_stretch=1.1, ver_stretch=1.1, rotation=20, brightness=0.1,
              contrast=0.4):
    rows, cols = image.shape[0], image.shape[1]

    # rotation
    M1 = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
    rot_plus_20 = cv2.warpAffine(image, M1, (cols, rows))

    M2 = cv2.getRotationMatrix2D((cols / 2, rows / 2), -1 * rotation, 1)
    rot_minus_20 = cv2.warpAffine(image, M2, (cols, rows))
    # Cropping part
    cropped = image[int(rows * per_cropped / 2):int(rows - rows * per_cropped / 2),
              int(cols * per_cropped / 2):int(cols - cols * per_cropped / 2)]

    # Stretching part
    hor_img = cv2.resize(image, dsize=None, fx=hor_stretch, fy=1)
    ver_img = cv2.resize(image, dsize=None, fx=1, fy=ver_stretch)

    ## brightness part and contrast part

    bright1 = cv2.addWeighted(image, 1, image, 0, (brightness) * 255)
    bright2 = cv2.addWeighted(image, 1, image, 0, (brightness * 2) * 255)
    contrast1 = cv2.addWeighted(image, 1 + contrast, image, 0, 0)
    contrast2 = cv2.addWeighted(image, 1 + contrast, image, 0, -255 * contrast)


    images = {
        "plus_20": rot_plus_20,
        "minus_20": rot_minus_20,
        "cropped": cropped,
        "hor_stretched": hor_img,
        "ver_stretched": ver_img,
        "bright1": bright1,
        "bright2": bright2,
        "contrast1": contrast1,
        "contrast2": contrast2,
        # "normalized": normalizedImg,
    }

    categories = [ "plus_20", "minus_20", "cropped", "hor_stretched", "bright1", "bright2", "ver_stretched",
                  "contrast1", "contrast2", "normalized"]

    random_index = random.sample(range(1, len(images)), 5)
    print(random_index)
    for index in random_index:
        images[categories[index]] = cv2.cvtColor(images[categories[index]], cv2.COLOR_BGR2GRAY)
    return images

MAIN_PATH ='./currency_dataset (copy)/'
#dirs  = ['fifty','fifty new','hundred','ten','twenty','two hundred','two thousand']
dirs = os.listdir(MAIN_PATH)
print(dirs)
for dir in dirs:
    path1 = os.path.join(MAIN_PATH,dir)
    print(path1)

    img_dirs = os.listdir(path1)

    for img_dir in img_dirs:
        path = os.path.join(path1,img_dir)


        img  = cv2.imread(path)
        path, ext = os.path.splitext(path)

        try:

            print(img.shape)
            images = operation(img)
            categories = [ "plus_20", "minus_20", "cropped", "hor_stretched", "bright1", "bright2",
                          "ver_stretched","contrast1", "contrast2"]
            for category in categories:
                img = images[category]
                path2 = path +'_'+category
                path2 = path2 +ext
                print(path2)
                cv2.imwrite(path2,img)
        except:
            print("exception")



