import torchvision
import torch
from tqdm import tqdm

from torchvision.models import resnet50, resnet34, vgg19_bn
from dataset import AFHQ_Dataset

from core import ModelStitcher
import dnnlib
import legacy
import utils
import random
import numpy as np
import torch.nn as nn

from settings import PATH_TO_AFHQ_WILD_TRAIN, PATH_TO_IMAGENET_TRAIN, SEED


def load_model(model_str):
    if model_str == "resnet50":
        return resnet50(pretrained=True).to("cuda").eval()
    elif ((model_str == "afhqwild.pkl") or (model_str == "imagenet")):
        model_str = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqwild.pkl"
        with dnnlib.util.open_url(model_str) as f:
            return WrapperStyleGAN2(legacy.load_network_pkl(f)['G_ema'].to("cuda").eval())
    elif model_str == "resnet34":
        return resnet34(pretrained=True).to("cuda").eval()
    elif model_str == "vgg19_bn":
        return vgg19_bn(pretrained=True).to("cuda").eval()
    elif model_str == "BigGAN":
        parser = utils.prepare_parser()
        parser = utils.add_sample_parser(parser)
        config = vars(parser.parse_args())
        config["resolution"] = utils.imsize_dict["I128_hdf5"]
        config["n_classes"] = utils.nclass_dict["I128_hdf5"]
        config["G_activation"] = utils.activation_dict["inplace_relu"]
        config["D_activation"] = utils.activation_dict["inplace_relu"]
        config["G_attn"] = "64"
        config["D_attn"] = "64"
        config["G_ch"] = 96
        config["D_ch"] = 96
        config["hier"] = True
        config["dim_z"] = 120
        config["shared_dim"] = 128
        config["G_shared"] = True
        config = utils.update_config_roots(config)
        config["skip_init"] = True
        config["no_optim"] = True
        config["device"] = "cuda"
        config["sample_random"] = True
        config["G_eval_mode"] = True
        config['experiment_name'] = "138k"

        model = __import__(config['model'])
        G = model.Generator(**config).cuda()

        state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0,
                        'best_IS': 0, 'best_FID': 999999, 'config': config}

        utils.load_weights(G if not (config['use_ema']) else None, None, state_dict,
                            config['weights_root'], "138k", config['load_weights'],
                            G if config['ema'] and config['use_ema'] else None,
                            strict=False, load_optim=False)

        return WrapperBigGAN(G).to("cuda").eval()



class WrapperStyleGAN2(nn.Module):
    def __init__(self, G):
        super().__init__()
        self.G = G

    def forward(self, batch_size):
        label = torch.zeros([batch_size, self.G.c_dim], device="cuda")
        z = torch.from_numpy(np.random.RandomState(random.randint(0, 100000000)).randn(batch_size, self.G.z_dim)).to("cuda")
        return self.G(z, label, truncation_psi=1, noise_mode='const')*0.5+0.5


class WrapperBigGAN(nn.Module):
    def __init__(self, G):
        super().__init__()
        self.G = G

    def forward(self, batch_size):
        z_, y_ = utils.prepare_z_y(batch_size, self.G.dim_z, 1000,
                                    device="cuda", fp16=False,
                                    z_var=1.0)
        z_.sample_()
        y_.sample_()
        G_z = self.G(z_, self.G.shared(y_))
        min_ = G_z.min()
        max_ = G_z.max()
        out = (G_z - min_) / (max_ - min_)
        return out



def train_afhq_stylegan2():
    layer_strings_from = ["layer1", "layer2", "layer3", "layer4"]
    layer_strings_to = ["G.synthesis.b128.conv0", "G.synthesis.b64.conv0", "G.synthesis.b32.conv0", "G.synthesis.b16.conv0"]
    for i in range(len(layer_strings_from)):
        layer_str_from = layer_strings_from[i]
        layer_str_to = layer_strings_to[i]
        torch.manual_seed(SEED)
        train(layer_str_from, layer_str_to, model_str_from="resnet50", model_str_to="afhqwild.pkl",
              dataset_name="afhq", n_epochs_to_train=30, batch_size=8)

def train_imagenet_stylegan2():
    layer_strings_from = ["layer1", "layer2", "layer3"]
    layer_strings_to = ["G.synthesis.b128.conv0", "G.synthesis.b64.conv0", "G.synthesis.b32.conv0"]
    for i in range(len(layer_strings_from)):
        layer_str_from = layer_strings_from[i]
        layer_str_to = layer_strings_to[i]
        torch.manual_seed(SEED)
        train(layer_str_from, layer_str_to, model_str_from="resnet50", model_str_to="imagenet",
              dataset_name="imagenet", batch_size=8)

def train_imagenet_BigGAN():
    layer_strings_from = ["layer1", "layer2", "layer3", "layer3"]
    layer_strings_to = ["G.blocks.3.0", "G.blocks.2.0", "G.blocks.1.0", "G.blocks.2.0"]
    for i in range(len(layer_strings_from)):
        layer_str_from = layer_strings_from[i]
        layer_str_to = layer_strings_to[i]
        torch.manual_seed(SEED)
        train(layer_str_from, layer_str_to, model_str_from="resnet50", model_str_to="BigGAN",
              dataset_name="imagenet", batch_size=32)


def get_layer_hook_point(model, layer_str):
    layer_hook_point = None
    for module in layer_str.split("."):
        if layer_hook_point == None:
            layer_hook_point = getattr(model, module)
        else:
            layer_hook_point = getattr(layer_hook_point, module)

    return layer_hook_point


def train(layer_str_from, layer_str_to, model_str_from, model_str_to, dataset_name, n_epochs_to_train=1,
          batch_size=8):
    model_from = load_model(model_str_from)
    model_to = load_model(model_str_to)

    if dataset_name == "imagenet":
        transforms = [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
        transforms = torchvision.transforms.Compose(transforms)
        dataset = torchvision.datasets.ImageFolder(PATH_TO_IMAGENET_TRAIN, transform=transforms)
    elif dataset_name == "afhq":
        transforms = [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
        transforms = torchvision.transforms.Compose(transforms)
        dataset = dataset = AFHQ_Dataset(transform=transforms, image_folder_path=PATH_TO_AFHQ_WILD_TRAIN)
    else:
        raise RuntimeError("dataset name: {} is not understood".format(dataset_name))


    train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)

    layer_hook_point_to = get_layer_hook_point(model_to, layer_str_to)
    network = ModelStitcher(network_from_segmentation=model_from, network_to_gan_generator=model_to,
                                      layer_hook_point_from_segmentation_encoder=eval("model_from." + layer_str_from),
                                      layer_hook_point_to_gan_decoder=layer_hook_point_to, rescale_activations=True)
    network = network.to("cuda")

    ckpt_path = "ckpts/" + model_str_from + "_to_" + model_str_to + "/" + layer_str_from + layer_str_to
    network.save_checkpoint(checkpoint_folder_path=ckpt_path)
    for epoch in range(n_epochs_to_train):
        i = 0
        for data in tqdm(train_loader, ascii=True):
            img = data[0].to("cuda")

            network.optimizer.zero_grad()
            loss = network.training_step(img, i)
            loss.backward()
            network.optimizer.step()

            i += 1
        network.save_checkpoint(checkpoint_folder_path=ckpt_path)


def main():
    train_afhq_stylegan2()
    train_imagenet_stylegan2()
    train_imagenet_BigGAN()


if __name__ == '__main__':
    main()



















