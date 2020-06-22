import os
import ntpath
# import pathlib  # TODO replace ntpath for pathlib


# from https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_all_files_recurse(rootdir):
    allfiles = []
    for root, directories, filenames in os.walk(rootdir):
        for filename in filenames:
            allfiles.append(os.path.join(root, filename))
    return allfiles


def filter_files(files, blacklist):
    prefiltered = []
    for f in files:
        todel = list(filter(lambda bl: bl in f, blacklist))
        if len(todel) == 0:
            prefiltered.append(f)
    return prefiltered