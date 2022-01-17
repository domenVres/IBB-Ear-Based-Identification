import os
import random


class TrainSplitter:
    def __init__(self, root, seed=7):
        self.root = root
        self.rnd = random.Random()
        self.rnd.seed(seed)

    def sort_train_data(self):
        """
        Function that sorts train images by their classes (person ID)
        :return: dictionary, where the keys are class IDs and values are lists of images belonging to that class
        """
        annot_file = self.root + "/annotations/recognition/ids.csv"
        f = open(annot_file, "r")

        sorted_imgs = {}

        for line in f.readlines():
            img, id = line.split(",")
            id = int(id.strip())

            # Skip test images
            if "test" in img: continue

            # Check if we read the class for the first time
            if id not in sorted_imgs:
                sorted_imgs[id] = []

            sorted_imgs[id].append(img.lstrip("train/"))

        return sorted_imgs

    def separate_val_data(self):
        imgs = self.sort_train_data()
        val_path = self.root + "/val"
        train_path = self.root + "/train"

        if not os.path.exists(val_path):
            os.makedirs(val_path)

        for img_list in imgs.values():
            # We move half of the images to val set (in case of odd number of images, we keep the additional one in
            # train set)
            val_size = len(img_list) // 2

            # Randomly shuffle the list and then the first val_size images will be moved to val set
            self.rnd.shuffle(img_list)
            val_imgs = img_list[:val_size]

            # Move files
            for img in val_imgs:
                try:
                    os.rename("/".join([train_path, img]), "/".join([val_path, img]))
                except:
                    print(f"Image {img} does not exist")


if __name__=="__main__":
    print("Separating perfectly detected ears")
    ts = TrainSplitter("perfectly_detected_ears")
    ts.separate_val_data()

    print("Separating Mask-R-CNN detecet ears")
    ts2 = TrainSplitter("Mask-R-CNN_detected_ears")
    ts2.separate_val_data()