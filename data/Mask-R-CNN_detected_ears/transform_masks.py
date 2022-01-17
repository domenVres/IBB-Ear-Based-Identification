import cv2
import os
from tqdm.auto import tqdm

import matplotlib.pyplot as plt


class MaskTransformer:

    def get_ear(self, mask, image):
        components = cv2.connectedComponentsWithStats(mask)
        detections = components[2]

        # Find the connected component with the largest area
        x, y, w, h, a_max = (None for i in range(5))
        for detection in detections[1:]:
            area = detection[4]
            if a_max is None or a_max < area:
                x = detection[0]
                y = detection[1]
                w = detection[2]
                h = detection[3]

        # Crop the ear rectangle from image
        cropped_image = image[y:y+h, x:x+w]

        return cropped_image


def retrieve_ears(transformer, data="train"):
    print("Transforming", data, "data ...")

    imlist = os.listdir(data+"_masks")

    for im_name in tqdm(imlist):
        mask = cv2.imread(data+"_masks/"+im_name, cv2.IMREAD_UNCHANGED)
        image = cv2.imread("../ears/"+data+"/"+im_name)

        cropped_image = transformer.get_ear(mask, image)
        cv2.imwrite(data+"/"+im_name, cropped_image)

    print("Done")


if __name__=="__main__":
    transformer = MaskTransformer()
    """mask = cv2.imread("test_masks/0007.png", cv2.IMREAD_UNCHANGED)
    image = cv2.imread("../ears/test/0007.png")
    cropped_image = transformer.get_ear(mask, image)
    cv2.imwrite("crop_test.png", cropped_image)
    plt.imshow(cropped_image)
    plt.show()"""

    retrieve_ears(transformer)
    retrieve_ears(transformer, data="test")
