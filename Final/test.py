# python test.py test classify model_name

import argparse
import glob, os, cv2, warnings
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.patches as patches
from datetime import datetime

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="parser")

parser.add_argument("test_dir", metavar="test", type=str, help='path to test root dir')
parser.add_argument("classify_dir", metavar="pre_processed", type=str, help='path to pre-processed root dir')
parser.add_argument("model_name", metavar="model_pth", type=str, help='path to train root dir')

# parse the arguments from standard input
args = parser.parse_args()

# input_path = r"./test/"
# output_path = r"classify/test/"

input_path = args.test_dir
device = 'cpu'
w, h = 182, 182

now = datetime.now()
# print("now =", now)
dt_string = now.strftime("%d_%m_%Y %H_%M_%S")
print("date and time =", dt_string)

output_path = args.classify_dir
output_path = os.path.join(output_path, dt_string)

saved_pth = args.model_name

print(args.test_dir)
print(args.classify_dir)
print(args.model_name)


def read_img(path):
    return cv2.imread(pic, cv2.IMREAD_UNCHANGED)


def resize_img(img, w, h):
    dim = (w, h)
    resized = cv2.resize(img, dim)
    return resized


def rotate_img(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def write_img(name, img):
    cv2.imwrite(name, img)


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def plt_image(img):
    plt.imshow(img)
    plt.show()


def plt_draw(x, y, width, height):
    ax = plt.gca()
    rect = patches.Rectangle((x, y), width, height, fill=False, color='red')
    ax.add_patch(rect)
    plt.show()


def detect(img_path):
    img = cv2.imread(img_path)
    faces = detector.detect_faces(img)
    return faces

sub_folder = '/test'
create_dir(output_path + sub_folder)
detector = MTCNN()

counter = 1
for pic in glob.glob(os.path.join(input_path, "*")):
    img = cv2.imread(pic)

    faces = detect(pic)

    img_name = pic.split("/")[-1]
    extension = img_name.split(".")[-1]

    if (len(faces) == 0):
        print('No : ', len(faces))
    else:
        print('Yes :', len(faces))
    for f in faces:
        new_img_name = str(counter) + "." + extension
        new_path = os.path.join(output_path+sub_folder, new_img_name)

        x, y, width, height = f['box']
        face = img[y:y + height, x:x + width, :]

        resized_img = resize_img(face, w, h)
        print(resized_img.shape)
        print(new_path)
        write_img(new_path, resized_img)
        counter += 1

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

testset = torchvision.datasets.ImageFolder(root=output_path, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# classes = ('Akhil', 'Ceepon', 'Gurpreet', 'Murali', 'Nikhil', 'Subhani')

classes = ('palla18130@iiitd.ac.in','ceepon18053@iiitd.ac.in','gurpreet18098@iiitd.ac.in',
            'kastala19132@iiitd.ac.in', 'kolla19123@iiitd.ac.in','subhani18117@iiitd.ac.in')

students = {'Akhil':'palla18130@iiitd.ac.in','Ceepon':'ceepon18053@iiitd.ac.in','Gurpreet':'gurpreet18098@iiitd.ac.in',
            'Murali':'kastala19132@iiitd.ac.in', 'Nikhil':'kolla19123@iiitd.ac.in', 'Subhani':'subhani18117@iiitd.ac.in'}

num_classes = 6

net = torchvision.models.resnet18(pretrained=False)
num_ftrs = net.fc.in_features

net.fc = nn.Linear(num_ftrs, num_classes)
# net.cuda()
net.to(device)

state_dict = torch.load(saved_pth)
net.load_state_dict(state_dict)

result = dict()
attended = []
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    # inputs, labels = inputs.cuda(), labels.cuda()
    inputs, labels = inputs.to(device), labels.to(device)

    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', predicted.item(), classes[predicted])
    attended.append(classes[predicted])

result['message'] = attended

print(result)
