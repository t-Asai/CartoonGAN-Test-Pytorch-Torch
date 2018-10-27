import torch
import os
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as vutils
from network.Transformer import Transformer
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="test_img")
parser.add_argument("--load_size", default=450)
parser.add_argument("--model_path", default="./pretrained_model")
parser.add_argument("--style", default="Hayao")
parser.add_argument("--output_dir", default="test_output")
parser.add_argument("--gpu", type=int, default=0)

opt = parser.parse_args()

valid_ext = [".jpg", ".png", ".gif"]


def convert_image(model, input_image):
    # resize image, keep aspect ratio
    h = input_image.size[0]
    w = input_image.size[1]
    ratio = h * 1.0 / w
    if ratio > 1:
        h = opt.load_size
        w = int(h * 1.0 / ratio)
    else:
        w = opt.load_size
        h = int(w * ratio)
    input_image = input_image.resize((h, w), Image.BICUBIC)
    input_image = np.asarray(input_image)
    # RGB -> BGR
    input_image = input_image[:, :, [2, 1, 0]]
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    # preprocess, (-1, 1)
    input_image = -1 + 2 * input_image
    with torch.no_grad():
        if opt.gpu > -1:
            input_image = Variable(input_image).cuda()
        else:
            input_image = Variable(input_image).float()
    # forward
    output_image = model(input_image)
    output_image = output_image[0]
    # BGR -> RGB
    output_image = output_image[[2, 1, 0], :, :]
    # deprocess, (0, 1)
    output_image = output_image.data.cpu().float() * 0.5 + 0.5
    return output_image


def save(image, name):
    vutils.save_image(image, name)


def jpg_to_gif(input_image, input_filename):
    images = []
    for filename in sorted(os.listdir("tmp")):
        ext = os.path.splitext(filename)[1]
        if ext not in valid_ext:
            continue
        image = Image.open("tmp/{}".format(filename)).convert("RGB")
        images.append(image)
    images[0].save("{dir}/{name}.gif".format(**{
        "dir": opt.output_dir,
        "name": "{}_{}".format(input_filename[:-4], opt.style)
    }),
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=input_image.info["duration"],
        loop=1)


def main():
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)

    # load pretrained model
    model = Transformer()
    model.load_state_dict(torch.load("{dir}/{name}".format(**{
        "dir": opt.model_path,
        "name": "{}_net_G_float.pth".format(opt.style)
    })))
    model.eval()

    if opt.gpu > -1:
        print("GPU mode")
        model.cuda()
    else:
        print("CPU mode")
        model.float()

    for filename in os.listdir(opt.input_dir):
        ext = os.path.splitext(filename)[1]
        if ext not in valid_ext:
            continue
        print(filename)
        # load image
        if ext == ".gif":
            if not os.path.exists("tmp"):
                os.mkdir("tmp")
            else:
                shutil.rmtree("tmp")
                os.mkdir("tmp")

            input_gif = Image.open(os.path.join(opt.input_dir, filename))
            for nframe in range(input_gif.n_frames):
                print("  {} / {}".format(nframe, input_gif.n_frames), end="\r")
                input_gif.seek(nframe)
                output_image = convert_image(
                    model, input_gif.split()[0].convert("RGB"))
                save(image=output_image,
                     name="tmp/{name}_{nframe:04d}.jpg".format(**{
                         "dir": opt.output_dir,
                         "name": "{}_{}".format(filename[:-4], opt.style),
                         "nframe": nframe
                     }))
            jpg_to_gif(input_gif, filename)
            shutil.rmtree("tmp")

        else:
            input_image = Image.open(os.path.join(
                opt.input_dir, filename)).convert("RGB")
            output_image = convert_image(model, input_image)
            # save
            save(image=output_image, name="{dir}/{name}.jpg".format(**{
                "dir": opt.output_dir,
                "name": "{}_{}".format(filename[:-4], opt.style)
            }))

    print("Done!")


if __name__ == "__main__":
    main()
