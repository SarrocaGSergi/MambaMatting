import os
import torch
import argparse
from PIL import Image
from models.SwinUamba import SwinUMamba
import torchvision.transforms as T


to_tensor = T.ToTensor()
to_pil = T.ToPILImage()


model = SwinUMamba(in_chans=3, out_chans=1, feat_size=[48, 96, 192, 384, 768], deep_supervision=True, hidden_size=768)
model.load_state_dict(torch.load("./best.pth"))
model = model.eval().cuda()


def create_out_directories(output_dir):
    com_dir = os.path.join(output_dir, "com")
    pha_dir = os.path.join(output_dir, "pha")
    out_dirs = [com_dir, pha_dir]
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)

def process_set(set_dir):
    image_list = os.listdir(os.path.join(set_dir, 'com'))
    for image in image_list:
        image = Image.open(os.path.join(set_dir, 'com', image))
        image = image.resize((256, 512))

        src = to_tensor(image)
        src = src.float().cuda()
        src = src.unsqueeze(0)
        with torch.no_grad():
            pha = model(src)
        x = 2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str or list[str], default='Brainstorm', choices=['Brainstorm', 'VideoMatte'],
                        required=False)
    args = parser.parse_args()

    output_dir = os.path.join("InferenceResults", args.dataset)
    gt_dir = os.path.join("/home/sergi-garcia/Projects/Finetunning/matting-data/HD/", args.dataset)
    create_out_directories(output_dir)
    sets = os.listdir(gt_dir)
    for set in sets:
        set_dir = os.path.join(gt_dir, set)
        process_set(gt_dir)



