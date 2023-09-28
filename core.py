import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
import os
import torchvision
from pathlib import Path



hook_output = None

def layer_hook(module, input_, output):
    global hook_output
    hook_output = output


def access_activations_forward_hook(x, forward_function, forward_hook_point):
    handle = forward_hook_point.register_forward_hook(layer_hook)
    with torch.no_grad():
        forward_function(*x)
    handle.remove()

    return hook_output.detach().cpu()


def cossim_gram_matrices_two_batches(batch_one, batch_two):
    batch, channels, _, _ = batch_one.shape
    flattened_one = batch_one.view(batch, channels, -1)
    flattened_two = batch_two.view(batch, channels, -1)
    gram_one = F.normalize(torch.matmul(flattened_one, torch.transpose(flattened_one, 1, 2)), p=2, dim=(1,2))
    gram_two = F.normalize(torch.matmul(flattened_two, torch.transpose(flattened_two, 1, 2)), p=2, dim=(1,2))
    return gram_one*gram_two


class ForwardHookSetChannelsToValue:
    def __init__(self, forward_hook_point, value_to_set_to, channels_to_set=[]):
        self.channels_to_set = channels_to_set
        self.value_to_set_to = value_to_set_to
        self.forward_hook_point = forward_hook_point

    def set_hook(self):
        return self.forward_hook_point.register_forward_hook(self.layer_hook)

    def layer_hook(self, module, input_, output):
        self.value_to_set_to = F.interpolate(self.value_to_set_to, size=(output.shape[-1], output.shape[-1]))

        output_clone = torch.clone(output)
        if len(self.channels_to_set) == 0:
            self.channels_to_set = [i for i in range(output.shape[1])]
        for channel in self.channels_to_set:
            if len(self.value_to_set_to.shape) <= 1:
                output_clone[:, channel] = self.value_to_set_to
            else:
                output_clone[:, channel] = self.value_to_set_to[:, channel]
        return output_clone


class ModelStitcher(nn.Module):
    def __init__(self, network_from_segmentation, network_to_gan_generator, layer_hook_point_from_segmentation_encoder, layer_hook_point_to_gan_decoder,
                rescale_activations=True,
                **kwargs):
        super().__init__(**kwargs)
        self.summary_writer = SummaryWriter()
        self.network_from_segmentation = network_from_segmentation
        self.network_to_gan_generator = network_to_gan_generator

        layer_hook_point_from_segmentation_encoder.register_forward_hook(self.hook_func_from)
        handle_hook_number_channels_to = layer_hook_point_to_gan_decoder.register_forward_hook(self.hook_func_to_for_number_channels)

        self.network_from_segmentation.to("cuda")
        self.network_from_segmentation.eval()
        self.network_to_gan_generator.to("cuda")
        self.network_to_gan_generator.eval()

        self.normalization = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        with torch.no_grad():
            self.network_from_segmentation(torch.zeros(1, 3, 224, 224).to("cuda"))
            self.network_to_gan_generator(1)

        handle_hook_number_channels_to.remove()
        layer_hook_point_to_gan_decoder.register_forward_hook(self.hook_func_to)

        n_ch_from = self.hook_out_from.shape[1]
        n_ch_to = self.hook_out_to_for_number_channels.shape[1]
        self.conv = torch.nn.Conv2d(n_ch_from, n_ch_to, 1, padding="same")

        self.size_from = self.hook_out_from.shape[-1]
        self.size_to = self.hook_out_to_for_number_channels.shape[-1]
        self.rescale_activations = rescale_activations

        self.configure_optimizers()


    def save_checkpoint(self, checkpoint_folder_path):
        weight_ckpt_path = checkpoint_folder_path + "/weights/"
        bias_ckpt_path = checkpoint_folder_path + "/bias/"
        Path(weight_ckpt_path).mkdir(parents=True, exist_ok=True)
        Path(bias_ckpt_path).mkdir(parents=True, exist_ok=True)

        assert (len(os.listdir(weight_ckpt_path)) == len(os.listdir(bias_ckpt_path)))
        n_files = len(os.listdir(weight_ckpt_path))

        torch.save(self.conv.weight, weight_ckpt_path + str(n_files).zfill(4) + ".pt")
        torch.save(self.conv.bias, bias_ckpt_path + str(n_files).zfill(4) + ".pt")


    def hook_func_from(self, module, input_, output):
        self.hook_out_from = output


    def hook_func_to_for_number_channels(self, module, input_, output):
        self.hook_out_to_for_number_channels = output


    def hook_func_to(self, module, input_, output):
        if self.rescale_activations:
            return F.interpolate(self.transferred, size=(self.size_to, self.size_to))
        else:
            return self.transferred

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.conv.parameters())


    def training_step(self, b_data, i):
        self.network_from_segmentation.eval()
        self.network_to_gan_generator.eval()

        batch_size = b_data.shape[0]
        tissue_image = b_data

        with torch.no_grad():
            self.network_from_segmentation(tissue_image)

        self.transferred = self.conv(self.hook_out_from)


        out_to = self.network_to_gan_generator(batch_size)

        out_to = F.interpolate(out_to, size=(224, 224), mode='bilinear')
        out_to = out_to
        out_to = torch.clamp(out_to, min=0.0, max=1.0)
        out_to = self.normalization(out_to)

        target = torch.clone(self.hook_out_from)
        self.network_from_segmentation(out_to)

        loss = torch.nn.functional.l1_loss(self.hook_out_from, target)

        self.summary_writer.add_scalar("loss", loss, i)

        return loss


















