import os
from os import listdir
from multiprocessing import Process

import logging
from wand.image import Image



"""

script for sequential editing the pictures

@author: Daniel Schnurpfeil
@version: 1.1.0
"""


class Edith(Process):

    def __init__(self, style, dir_size=None):
        super().__init__()
        self.style = style
        self.dir_size = len(listdir("../../original_data/" + self.style))

    def run(self) -> None:
        super().run()
        # bez pripony
        folder = self.style
        purpose = ['train', 'val']

        #
        path = '../../ready_data/' + purpose[0] + '/' + folder + '/'
        logging.info(folder + " start")
        directory = listdir("../../original_data/" + folder)
        for i in range(int(self.dir_size * 0.80)):
            process_img("../../original_data/" + folder + "/" + directory[i], path, directory[i])

            # logging.info(self.style + " current picture: " + i.__str__())

        path = '../../ready_data/' + purpose[1] + '/' + folder + '/'
        logging.info(folder + " val")
        for i in range(
                int(self.dir_size * 0.80),
                int(self.dir_size)):
            process_img("../../original_data/" + folder + "/" + directory[i], path, directory[i])
            # logging.info(self.style + " current picture: " + i.__str__())
        logging.info(folder + " end")


def process_img(img, path, file_name):
    # rotated_names = []
    # all_names = []
    img = img.replace(".jpg", "")
    path += file_name.replace(".jpg", "")

    new_names = [copy(img, path),
                 mirror(img, path),
                 blur(img, path),
                 noise(img, path)
                 ]

    # for name in new_names:
    #     rotated_names.append(rotate_left(name, name))
    #     rotated_names.append(rotate_right(name, name))

    # for name in rotated_names:
    #     all_names.append(crop_left(name, name))
    #     all_names.append(crop_right(name, name))

    # for name in all_names:
    #     prepare(name, name)

    # for name in rotated_names:
    #     prepare(name, name)

    for name in new_names:
        prepare(name, name)


def prepare(filename, destination):
    """

    :param destination: output
    :param filename: input
    :return: new name
    """
    with Image(filename=filename + ".jpg") as i:
        i.resize(500, 500)  # 400 - 600 px strany
        i.save(filename=destination + ".jpg")


def copy(filename, destination):
    name = str(destination)
    with Image(filename=filename + ".jpg") as img:
        # print(name)
        with img.clone() as i:
            i.save(filename=name + ".jpg")

    return name


def rotate_right(filename, destination):
    """
    rotate_right by 10 degrees
    :param destination: output
    :param filename: input
    :return: new name
    """
    name = str(destination + "_rot_right")
    with Image(filename=filename + ".jpg") as img:
        # print(name)
        with img.clone() as i:
            i.rotate(10)

            width, heigth = aspect(float(img.width), float(img.height),
                                   float(i.width), float(i.height))
            # print(width, heigth)

            i.crop(width=int(width), height=int(heigth), gravity='center')
            i.save(filename=name + ".jpg")
    return name


def rotate_left(filename, destination):
    """
    rotate_left by 10 degrees
    :param destination: output
    :param filename: input
    :return: new name
    """
    name = str(destination + "_rot_left")
    with Image(filename=filename + ".jpg") as img:
        # print(name)
        with img.clone() as i:
            i.rotate(-10)

            width, heigth = aspect(float(img.width), float(img.height),
                                   float(i.width), float(i.height))

            i.crop(width=int(width), height=int(heigth), gravity='center')
            i.save(filename=name + ".jpg")
    return name


def aspect(width, height, aspect_x, aspect_y):
    """
    calculates picture to crops to fit after rotate
    :param width: ...
    :param height: ...
    :param aspect_x: rotated width
    :param aspect_y: rotated height
    :return: new width, new height
    """
    old_ratio = width / height
    new_ratio = aspect_x / aspect_y
    if old_ratio < 1:
        old_ratio = height / width
        new_ratio = aspect_y / aspect_x
    reduction_x = aspect_x - width
    reduction_y = aspect_y - height
    if abs(old_ratio - new_ratio) < 0.1:
        if new_ratio > old_ratio:
            return width / new_ratio - reduction_x, \
                   height / new_ratio - reduction_y  # same width, shorter height
        else:
            return width / new_ratio - reduction_x, \
                   height / new_ratio - reduction_y  # shorter width, same height

    elif new_ratio > old_ratio:
        return width / new_ratio, height / new_ratio  # same width, shorter height
    else:
        return width / new_ratio, height / new_ratio  # shorter width, same height


def crop_left(filename, destination):
    """
    crops top or left
    :param destination: output
    :param filename: input
    :return: new name
    """
    name = str(destination + "_crop_left")
    with Image(filename=filename + ".jpg") as img:
        # print(name)
        with img.clone() as i:
            if img.width > img.height:
                i.crop(left=int(img.width * 0.1))
            else:
                i.crop(top=int(img.height * 0.1))
            i.save(filename=name + ".jpg")
    return name


def crop_right(filename, destination):
    """
    crops bottom or right
    :param destination: output
    :param filename: input
    :return: new name
    """
    name = str(destination + "_crop_right")
    with Image(filename=filename + ".jpg") as img:
        # print(name)
        with img.clone() as i:
            if img.width > img.height:
                i.crop(right=int(img.width * 0.9))
            else:
                i.crop(bottom=int(img.height * 0.9))
            i.save(filename=name + ".jpg")
    return name


def mirror(filename, destination):
    """
    makes mirror
    :param destination: output
    :param filename: input
    :return: new name
    """
    name = str(destination + "_mirror")
    with Image(filename=filename + ".jpg") as img:
        # print(name)
        with img.clone() as i:
            i.flop()
            i.save(filename=name + ".jpg")
    return name


def noise(filename, destination):
    """
    adds noise
    :param destination: output
    :param filename: input
    :return: new name
    """
    name = str(destination + "_noise")
    with Image(filename=filename + ".jpg") as img:
        # print(name)
        with img.clone() as i:
            i.modulate(brightness=60.0, saturation=35, hue=100)
            i.noise('multiplicative_gaussian')
            i.save(filename=name + ".jpg")
    return name


def blur(filename, destination):
    """
    makes motion blur
    :param destination: output
    :param filename: input
    :return: new name
    """
    name = str(destination + "_blur")
    with Image(filename=filename + ".jpg") as img:
        # print(name)
        with img.clone() as i:
            i.motion_blur(radius=5, sigma=10, angle=2)
            i.save(filename=name + ".jpg")
    return name


def make_folders():
    import os

    from training_scripts.preprocesing_imgs.checker.loader import load_styles

    if "ready_data" in os.listdir("../../") and os.path.isdir("../../ready_data"):
        for style in load_styles("styles_list.txt"):
            os.path.isdir("../../ready_data/train/" + style)
            os.path.isdir("../../ready_data/val/" + style)
        return
    else:
        os.mkdir("../../ready_data")
        os.mkdir("../../ready_data/train/")
        os.mkdir("../../ready_data/val/")
        for style in load_styles("styles_list.txt"):
            os.mkdir("../../ready_data/train/" + style)
            os.mkdir("../../ready_data/val/" + style)


def main():
    logging.basicConfig(format='[%(levelname)s] - %(asctime)s - %(message)s',
                        level='INFO')

    make_folders()

    # set MAGICK_HOME /opt/homebrew/Cellar/imagemagick@6/6.9.12-86_1
    # if is magic not found

    minimum = 0  # magic number
    Edith('art_nouveau', minimum).start()
    Edith('asian', minimum).start()
    Edith('baroque', minimum).start()
    Edith('brutalist', minimum).start()
    Edith('cubism', minimum).start()
    Edith('functionalism', minimum).start()
    Edith('gothic', minimum).start()
    Edith('islamic', minimum).start()
    Edith('renaissance', minimum).start()
    Edith('romanesque', minimum).start()


if __name__ == '__main__':
    main()
