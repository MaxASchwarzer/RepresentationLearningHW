import os
import random
import argparse

split = 0.8

for root, dirs, files in os.walk('trainset'):
    if len(dirs) == 0:
        print(root)
        random.shuffle(files)
        train = files[:int(0.8*len(files))]
        validate = files[int(0.8*len(files)):]

        label = root.split("/")[1]
        validation_path = os.path.join("validationset", label)

        if not os.path.exists(validation_path):
            os.makedirs(validation_path)

        for filename in validate:
            os.rename(os.path.join(root, filename),
                      os.path.join(validation_path, filename))
