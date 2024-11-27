import glob, os, cv2


def read_img(path):
    return cv2.imread(pic, cv2.IMREAD_UNCHANGED)


def resize_img(img, width, height):
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def rotate_img(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def write_img(name, img):
    cv2.imwrite(name, img)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


path = "../Dataset/input/sample"
out = "../Dataset/output/"
inp = path.split("/")

result_folder = os.path.join(out, inp[-1])

create_dir(result_folder)

for cls in glob.glob(os.path.join(path, "*")):
    cls_name = cls.split("\\")[1]
    cls_folder = os.path.join(result_folder, cls_name)

    create_dir(cls_folder)

    counter = 0
    for pic in glob.glob(os.path.join(cls, "*")):
        img = cv2.imread(pic, cv2.IMREAD_UNCHANGED)

        width, height = 1280, 960
        resized_img = resize_img(img, width, height)
        rotated_img = rotate_img(resized_img)

        img_name = pic.split("\\")[2]
        extension = img_name.split(".")[1]
        new_img_name = cls_name + "_" + str(counter) + "." + extension
        new_path = os.path.join(cls_folder, new_img_name)
        counter += 1

        write_img(new_path, rotated_img)
