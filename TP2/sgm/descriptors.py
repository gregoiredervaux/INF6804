import numpy as np
from skimage.feature import hog, BRIEF


def dense_hog(img, dsc_size, num_orientations):

    # le but est d'obtenir un descripteur de dimension définie pour chaque pixel (HOG Dense)
    rows, cols = img.shape

    # on créé une matrice vide pour contenir nos descripteurs de la taille de notre image
    img_hog = np.zeros((rows, cols, num_orientations))

    for i in range(rows):
        for j in range(cols):

            # pour chaque pixel, on détèrmine un cadre englobant de 8 pixels
            # si ce n'est pas possible (on est sur les bords de l'image) on réduit le cadre
            # on applique HOG sur ce cadre et on stock le vecteur aux coordonnées correspondantes
            local_img = img[max(0, i - int(dsc_size / 2)):min(rows, i + int(dsc_size / 2)), max(0, j - int(dsc_size / 2)):min(cols, j + int(dsc_size / 2))]
            img_hog[i,j,:] = hog(local_img, orientations=num_orientations, pixels_per_cell=(local_img.shape[0], local_img.shape[1]), cells_per_block=(1,1))
    return img_hog

def apply_hog(left, right, descriptor_size, num_orientations):
    """
    computes HOG descriptor on both images.
    :param left: left image.
    :param right: right image.
    :param descriptor_size: number of pixels in a hog cell.
    :param num_orientations: number of HOG orientations.
    :return: (H x W x M) array, H = height, W = width and M = num_orientations, of type np.float32.
    """
    # on applique dense hog sur chaque image
    return (dense_hog(left, descriptor_size, num_orientations), dense_hog(right, descriptor_size, num_orientations))


def apply_brief(left, right, descriptor_size, num_elements):
    """
    computes BRIEF descriptor on both images.
    :param left: left image.
    :param right: right image.
    :param descriptor_size: size of window of the BRIEF descriptor.
    :param num_elements: length of the feature vector.
    :return: (H x W) array, H = height and W = width, of type np.int64
    """

    # on génère la liste des points clefs. Comme on utilise une approche dense, chaque pixel est un point clef
    pixels_coordonates = [[[j, i] for i in range(left.shape[1])] for j in range(left.shape[0])]
    # la liste des points clefs doit être sous la forme d'une liste de coordonnée
    keypoints = np.array(pixels_coordonates).reshape((left.shape[1] * left.shape[0]), 2)

    # on utilise Brief du package skimage pour obtenir les descripteurs de chaque pixel
    extractor = BRIEF(descriptor_size=num_elements, patch_size=descriptor_size, mode='normal')

    extractor.extract(left, keypoints)
    descriptors1 = extractor.descriptors

    extractor.extract(right, keypoints)
    descriptors2 = extractor.descriptors

    # Nous avons maitenant un vecteur qui décrit notre image pixel par pixel.
    # Pour appliquer SGM, nous devons reconstruire la forme initiale de l'image.
    # Selon la taille de la fenetre d'analyse (descriptor_size) les bords de l'images n'ont pas pu être traité,
    # La taille de l'image est donc réduite de la taille du descritpeur
    descriptors1.resize((left.shape[0] - descriptor_size + 1, left.shape[1] - descriptor_size + 1, num_elements), refcheck=False)
    descriptors2.resize((left.shape[0] - descriptor_size + 1, left.shape[1] - descriptor_size + 1, num_elements), refcheck=False)

    # Ici, la taille de descripteur est de 128 bit.
    # Le problème est que SGM et la distance de hamming prennent en entrée un entier
    # nous devons donc transformer 128 bit en un entier.
    # numpy et le traitement qui suit (distance de hamming) n'admet pas d'entier suppérieur à 64 bit.
    # Nous devons donc réduire la dimension de notre descripteur.
    # nous avons testé plusieurs valeures de réduction, et avons choisi 1 bit sur 20.
    # C'est la plus grande réduction de dimension testée sans altération des résultats obtenus sur l'image des cones.

    concat_desc1 = np.apply_along_axis(lambda list: int(''.join([str(int(v)) if i % 20 == 0 else '' for i, v in enumerate(list)])), 2, descriptors1)
    concat_desc2 = np.apply_along_axis(lambda list: int(''.join([str(int(v)) if i % 20 == 0 else '' for i, v in enumerate(list)])), 2, descriptors2)

    # Enfin, comme il est requis de passer une matrice de taille égale à l'image de départ, nous allons rajouter des 0
    # là ou l'information manque, en bordure d'image.
    padding_pattern = (descriptor_size // 2 - 1, descriptor_size // 2)

    padded_descr1 = np.pad(concat_desc1, (padding_pattern, padding_pattern), 'constant')
    padded_descr2 = np.pad(concat_desc2, (padding_pattern, padding_pattern), 'constant')

    return (padded_descr1, padded_descr2)