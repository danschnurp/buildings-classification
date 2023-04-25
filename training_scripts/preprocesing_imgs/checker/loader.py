"""

loader

@author: Daniel Schnurpfeil
@version: 1.0.0

"""


def load_styles(styles):
    """
    lad styles from txt
    :return: list of styles
    """
    dir_list = []
    with open(styles) as txt:
        for line in txt:
            dir_list.append(line.split(" ")[0].lower())

    dir_list = sorted(dir_list)
    print(dir_list)
    return dir_list
