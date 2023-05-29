import os
import shutil

# this is used to take the last number underscore and .jpg out from the string
def breed(st):
    # Find the indices of the first and second underscore
    basename, extension = os.path.splitext(os.path.basename(st))
    basename = basename[:-len("_1")]
    basename = ''.join(char for char in basename if not char.isdigit())
    basename = basename.rstrip("_")
    return basename


# where the images where initialy
DIR = "D:\study_ml\data_images_v2\images"
# where i wanted the images to go
DEST = "D:\study_ml\data_images_v2\cat_breeds"

# check if the dir we want exist
if os.path.exists(DIR):
    # loop through all the images
    for file in os.listdir(DIR):
        # delete all the photos that are dogs because dog breeds started with lower case
        if not file[0].isupper():
            os.remove(f"{DIR}\{file}")
        # delete any files that are not a photo
        if not file.__contains__(".jpg"):
            os.remove(f"{DIR}\{file}")
        # move all the cat breeds to their intended file
        shutil.move(f"{DIR}\{file}", f"{DEST}\{breed(file)}\{file}")
else:
    print("file does not exist")
