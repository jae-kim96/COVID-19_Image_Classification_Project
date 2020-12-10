from PIL import Image
import glob
import os
from random import randrange

# Resize the image to a set pixel width and height
def resizeIm(im):
    w, h = im.size
    #print('Original Sizes are w = {}, h = {}'.format(w, h))
    if w == 224 and h == 224:
        return im
    else:
        return im.resize((224, 224))

# Introduce Random Flip to the images in order to augment the data
def randomFlip(im):
    num = randrange(0, 1)
    if num == 0:
        return im
    else:
        return im.transpose(Image.FLIP_LEFT_RIGHT)


def main():
    covidPath = '../images/preprocessed/covid19'
    normalPath = '../images/preprocessed/normal'
    pneumoniaPath = '../images/preprocessed/pneumonia'

    allPaths = [covidPath, normalPath, pneumoniaPath]

    newCovidPath = '../images/gray/processed/covid19/'
    newNormalPath = '../images/gray/processed/normal/'
    newPneumoniaPath = '../images/gray/processed/pneumonia/'

    newPaths = [newCovidPath, newNormalPath, newPneumoniaPath]
    
    types = ['covid19', 'normal', 'pneumonia']
    
    # numInputImages = 150 # Number of Images desired for the input
    print('-----------INITIATING RESIZING AND RANDOM FLIPS-----------')
    for i in range(len(allPaths)):
        path = allPaths[i]
        print('Working on {} Image Set'.format(types[i]))
        count = 1
        for filename in os.listdir(path):
            fullname = path + '/' + filename
            if fullname.endswith('.png') or filename.endswith('.jpeg'):
                img = Image.open(fullname)
                resized = resizeIm(img)
                flipped = randomFlip(resized)
                newPath = newPaths[i]
                # gray = flipped.convert('LA')
                final_im = flipped.convert('L')
                newName = newPath + types[i] + '-processed-' + str(count) + '.jpeg'
                final_im.save(newName)
            count += 1
        print('Finished {} Image Set: {} Images Processed'.format(types[i], count - 1))
        print('........')

    print('-----------FINISHED RESIZING AND RANDOM FLIPS-----------')




if __name__ == "__main__":
    main()