import os
from collections import defaultdict
import random

def get_images_name():
    divided_images=defaultdict(list)
    for single_file in os.listdir("./data"):
        if("GT" not in single_file and ".bmp" in single_file):
            divided_images[single_file[0]].append(single_file)

    test_set=[]
    training_set=[]
    for index in divided_images:
        elem=random.choice(divided_images[index])
        divided_images[index].remove(elem)

        dictionary_test={}
        dictionary_test["name"]=elem
        test_set.append(dictionary_test)

        for elem_train in divided_images[index]:
            dictionary_train={}
            dictionary_train["name"]=elem_train
            training_set.append(dictionary_train)

    return training_set,test_set