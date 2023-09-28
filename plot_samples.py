
import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision.utils import save_image
from train import load_model, get_layer_hook_point
from core import access_activations_forward_hook, ForwardHookSetChannelsToValue
from settings import CONV_CHECKPOINT_PATH, SEED
from data_utils import get_dataset, get_transform_from_gan_output_to_feature_extractor_input


def vis_gan_activations(x, model_from, hook_point_from, conv_weight, conv_bias, layer_hook_point_to, model_to):
    activations_shape_like_to = access_activations_forward_hook([x.shape[0]], model_to, layer_hook_point_to)
    size_to = activations_shape_like_to.shape[-1]

    batch_size = x.shape[0]
    activation_from = access_activations_forward_hook([x], model_from.forward, hook_point_from)
    transferred_activation = F.conv2d(activation_from.to("cuda"), conv_weight.to("cuda"), conv_bias.to("cuda"))
    transferred_activation = F.interpolate(transferred_activation, size=(size_to, size_to))
    hook = ForwardHookSetChannelsToValue(forward_hook_point=layer_hook_point_to, value_to_set_to=transferred_activation.to("cuda")).set_hook()
    with torch.no_grad():
        outputs = model_to(batch_size)

        input_transfer = torch.clamp(outputs, min=0.0, max=1.0)
    hook.remove()

    return input_transfer


def plot_images(model_str_from, model_str_to, layers_from, layers_to, conv_ckpt_indices, images,
                image_save_folder):
    model_from = load_model(model_str_from)
    model_to = load_model(model_str_to)

    normalized_cropped_imgs = get_transform_from_gan_output_to_feature_extractor_input()(images)

    torch.manual_seed(SEED)

    normalized_cropped_imgs = normalized_cropped_imgs.to("cuda")

    Path(image_save_folder).mkdir(parents=True, exist_ok=True)
    save_image(images, image_save_folder+"original_imgs.png")

    for i, layer_from in enumerate(layers_from):
        layer_to = layers_to[i]
        conv_ckpt_index = conv_ckpt_indices[i]
        ckpt_path = CONV_CHECKPOINT_PATH + model_str_from + "_to_" + model_str_to + "/" + layer_from + layer_to
        conv_weight = torch.load(ckpt_path + "/weights/" + str(conv_ckpt_index).zfill(4) + ".pt").to("cuda")
        conv_bias = torch.load(ckpt_path + "/bias/" + str(conv_ckpt_index).zfill(4) + ".pt").to("cuda")
        imgs = vis_gan_activations(normalized_cropped_imgs, model_from, getattr(model_from, layer_from), conv_weight, conv_bias,
                                                    get_layer_hook_point(model_to, layer_to),
                                                    model_to)

        save_image(torch.tensor(imgs), image_save_folder+"sample" + model_str_from + model_str_to + layer_from + layer_to + ".png")


def main():
    dataset = get_dataset("imagenet_val_500")
    imgs_imagenet = []
    for i in range(8):
        processed_img, original_img = dataset[-i]
        imgs_imagenet.append(original_img)
    imgs_imagenet = torch.stack(imgs_imagenet, dim=0)

    image_save_folder_imagenet = "images_imagenet_val/"
    plot_images(model_str_from="resnet50", model_str_to="BigGAN", layers_from=["layer1", "layer2", "layer3", "layer3"],
          layers_to=["G.blocks.3.0", "G.blocks.2.0", "G.blocks.1.0", "G.blocks.2.0"], conv_ckpt_indices=[1, 1, 1, 1],
          images=imgs_imagenet, image_save_folder=image_save_folder_imagenet)

    plot_images(model_str_from="resnet50", model_str_to="imagenet", layers_from=["layer1", "layer2", "layer3", "layer3"],
          layers_to=["G.synthesis.b128.conv0", "G.synthesis.b64.conv0", "G.synthesis.b32.conv0", "G.synthesis.b64.conv0"], conv_ckpt_indices=[1, 1, 1, 1],
          images=imgs_imagenet, image_save_folder=image_save_folder_imagenet)

    plot_images(model_str_from="resnet50", model_str_to="afhqwild.pkl", layers_from=["layer1", "layer2", "layer3", "layer3"],
          layers_to=["G.synthesis.b128.conv0", "G.synthesis.b64.conv0", "G.synthesis.b32.conv0", "G.synthesis.b64.conv0"], conv_ckpt_indices=[20, 27, 29, 29],
          images=imgs_imagenet, image_save_folder=image_save_folder_imagenet)


    dataset = get_dataset("afhq")
    imgs_afhq = []
    for i in range(8):
        processed_img, original_img = dataset[i]
        imgs_afhq.append(original_img)
    imgs_afhq = torch.stack(imgs_afhq, dim=0)

    image_save_folder_afhq = "images_afhq/"
    plot_images(model_str_from="resnet50", model_str_to="BigGAN", layers_from=["layer1", "layer2", "layer3", "layer3"],
          layers_to=["G.blocks.3.0", "G.blocks.2.0", "G.blocks.1.0", "G.blocks.2.0"], conv_ckpt_indices=[1, 1, 1, 1],
          images=imgs_afhq, image_save_folder=image_save_folder_afhq)

    plot_images(model_str_from="resnet50", model_str_to="imagenet", layers_from=["layer1", "layer2", "layer3", "layer3"],
          layers_to=["G.synthesis.b128.conv0", "G.synthesis.b64.conv0", "G.synthesis.b32.conv0", "G.synthesis.b64.conv0"], conv_ckpt_indices=[1, 1, 1, 1],
          images=imgs_afhq, image_save_folder=image_save_folder_afhq)

    plot_images(model_str_from="resnet50", model_str_to="afhqwild.pkl",
                layers_from=["layer1", "layer2", "layer3", "layer3", "layer3", "layer3", "layer3", "layer4"],
                layers_to=["G.synthesis.b128.conv0", "G.synthesis.b64.conv0", "G.synthesis.b8.conv0", "G.synthesis.b16.conv0",
                           "G.synthesis.b32.conv0", "G.synthesis.b64.conv0", "G.synthesis.b128.conv0", "G.synthesis.b16.conv0"],
                           conv_ckpt_indices=[20, 27, 29, 29, 29, 29, 29, 30],
                images=imgs_afhq, image_save_folder=image_save_folder_afhq)




if __name__ == '__main__':
    main()
