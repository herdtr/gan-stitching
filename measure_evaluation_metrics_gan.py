import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import time
from tqdm import tqdm
import numpy as np
import os
from train import load_model, get_layer_hook_point
from core import (
    access_activations_forward_hook,
    ForwardHookSetChannelsToValue,
    cossim_gram_matrices_two_batches,
)
from settings import CONV_CHECKPOINT_PATH, SEED

from data_utils import get_dataset, get_transform_from_gan_output_to_feature_extractor_input


def run_metrics(test_activation, original_activation):
    cossim = torch.mean(
        torch.nn.functional.cosine_similarity(
            test_activation, original_activation, dim=1
        ),
        dim=[1, 2],
    )
    l1_loss = torch.mean(
        torch.nn.functional.l1_loss(
            test_activation, original_activation, reduction="none"
        ),
        dim=[1, 2, 3],
    )
    cossim_gram_matrices = torch.sum(
        cossim_gram_matrices_two_batches(test_activation, original_activation),
        dim=[1, 2],
    )

    return {
        "cosine_similarity": cossim,
        "L1_Loss": l1_loss,
        "cossim_gram_matrices": cossim_gram_matrices,
    }


def reconstruct_input(
    x,
    model_from,
    hook_point_from,
    conv_weight,
    conv_bias,
    size_to,
    layer_hook_point_to,
    model_to,
    use_stitching_convolution,
):
    batch_size = x.shape[0]
    activation_from = access_activations_forward_hook(
        [x], model_from.forward, hook_point_from
    )
    if use_stitching_convolution:
        transferred_activation = F.conv2d(
            activation_from.to("cuda"), conv_weight.to("cuda"), conv_bias.to("cuda")
        )
    else:
        transferred_activation = activation_from.to("cuda")
    transferred_activation = F.interpolate(
        transferred_activation, size=(size_to, size_to)
    )
    hook = ForwardHookSetChannelsToValue(
        forward_hook_point=layer_hook_point_to,
        value_to_set_to=transferred_activation.to("cuda"),
    ).set_hook()
    with torch.no_grad():
        outputs = model_to(batch_size)
        outputs = outputs
        input_transfer = torch.clamp(outputs, min=0.0, max=1.0)
    hook.remove()

    return input_transfer


def eval_(
    layer_str_from,
    layer_str_to,
    model_str_from,
    model_str_to,
    list_model_str_test,
    layer_hooks_test,
    dataset,
    first_ckpt_only=False,
    use_stitching_convolution=True,
):
    model_from = load_model(model_str_from)
    model_to = load_model(model_str_to)

    hook_point_from = getattr(model_from, layer_str_from)

    #dataset = get_dataset(dataset_name)
    transforms_gan_to_seg = get_transform_from_gan_output_to_feature_extractor_input()

    data_loader = DataLoader(
        dataset, batch_size=50, num_workers=8, pin_memory=False, shuffle=False
    )

    ckpt_path = (
        CONV_CHECKPOINT_PATH
        + model_str_from
        + "_to_"
        + model_str_to
        + "/"
        + layer_str_from
        + layer_str_to
    )

    torch.manual_seed(SEED)
    random.seed(SEED)

    layer_hook_point_to = get_layer_hook_point(model_to, layer_str_to)

    activations_shape_like_to = access_activations_forward_hook(
        [1], model_to, layer_hook_point_to
    )
    size_to = activations_shape_like_to.shape[-1]

    print("load models...")
    test_models = []
    for i, model_str in enumerate(list_model_str_test):
        test_model = load_model(model_str)
        test_models.append(test_model)

    all_result_metrics = {test_model_str: {} for test_model_str in list_model_str_test}

    start = time.time()
    files_weights = sorted(os.listdir(ckpt_path + "/weights/"))
    files_bias = sorted(os.listdir(ckpt_path + "/bias/"))
    assert len(files_weights) == len(files_bias)

    for index in range(len(files_weights)):
        conv_weight = torch.load(ckpt_path + "/weights/" + files_weights[index]).to(
            "cuda"
        )
        conv_bias = torch.load(ckpt_path + "/bias/" + files_bias[index]).to("cuda")
        result_metrics = {test_model_str: {} for test_model_str in list_model_str_test}
        for data in tqdm(data_loader, disable=False, ascii=True):
            x = data[0].to("cuda")

            input_transfer = reconstruct_input(
                x,
                model_from,
                hook_point_from,
                conv_weight,
                conv_bias,
                size_to,
                layer_hook_point_to,
                model_to,
                use_stitching_convolution,
            )

            input_transfer_transformed = transforms_gan_to_seg(input_transfer)

            for i, model_str in enumerate(list_model_str_test):
                test_model = test_models[i]

                activation_transfer = access_activations_forward_hook(
                    [input_transfer_transformed],
                    test_model.forward,
                    eval("test_model" + layer_hooks_test[i]),
                ).cuda()
                activation_original = access_activations_forward_hook(
                    [x], test_model.forward, eval("test_model" + layer_hooks_test[i])
                ).to("cuda")

                metrics_step = run_metrics(activation_transfer, activation_original)
                for key in metrics_step:
                    if not key in result_metrics[model_str].keys():
                        result_metrics[model_str][key] = []
                    for value in metrics_step[key]:
                        result_metrics[model_str][key].append(value.cpu())

        for i, model_str in enumerate(list_model_str_test):
            for key in result_metrics[model_str]:
                if not key in all_result_metrics[model_str].keys():
                    all_result_metrics[model_str][key] = {"mean": [], "std": []}
                mean = torch.mean(
                    torch.from_numpy(np.array(result_metrics[model_str][key]))
                )
                std = torch.std(
                    torch.from_numpy(np.array(result_metrics[model_str][key]))
                )
                all_result_metrics[model_str][key]["mean"].append(mean)
                all_result_metrics[model_str][key]["std"].append(std)

        if first_ckpt_only:
            break

    time_taken = time.time() - start

    print("--------------------------------------------------------------------")
    print("--------------------------------------------------------------------")
    print("from layer: {}, to layer: {}".format(layer_str_from, layer_str_to))
    print("use stitching convolution: {}".format(use_stitching_convolution))

    for i, model_str in enumerate(list_model_str_test):
        print("model str: {}".format(model_str))
        for key in all_result_metrics[model_str]:
            print(
                "mean "
                + key
                + " value: {}".format(all_result_metrics[model_str][key]["mean"])
            )
            print(
                "std "
                + key
                + " value: {}".format(all_result_metrics[model_str][key]["std"])
            )
        print("--------------------------------------------------------------------")

    print("--------------------------------------------------------------------")
    print("--------------------------------------------------------------------")

    return time_taken


def main():
    evaluation_on_afhq()
    print("--------------------------------------------------------------------")
    print("--------------------------------------------------------------------")
    print("--------------------------for imagenet------------------------------")
    print("--------------------------------------------------------------------")
    print("--------------------------------------------------------------------")
    evaluation_on_imagenet()


def evaluation_on_imagenet():

    layer_strings = (
        ["layer1", "layer2", "layer3", "layer1", "layer2", "layer3"]
        + ["layer1", "layer2", "layer3", "layer3"]
    )
    model_to = (
        ["afhqwild.pkl" for i in range(3)]
        + ["imagenet" for i in range(3)]
        + ["BigGAN" for i in range(4)]
    )
    layer_strings_to = (
        [
            "G.synthesis.b128.conv0",
            "G.synthesis.b64.conv0",
            "G.synthesis.b32.conv0",
            "G.synthesis.b128.conv0",
            "G.synthesis.b64.conv0",
            "G.synthesis.b32.conv0",
        ] + ["G.blocks.3.0", "G.blocks.2.0", "G.blocks.1.0", "G.blocks.2.0"]
    )
    vgg_test_layers = (
        [
            ".features[12]",
            ".features[25]",
            ".features[38]",
            ".features[12]",
            ".features[25]",
            ".features[38]",
        ]
        + [
            ".features[12]",
            ".features[25]",
            ".features[38]",
        ]
        + [
            ".features[12]",
            ".features[25]",
            ".features[38]",
            ".features[38]",
        ]
    )
    use_stitching_convolution = [True, True, True] + [True for i in range(7)]

    for i, layer_str in enumerate(layer_strings):
        torch.manual_seed(42)
        layer_str_to = layer_strings_to[i]
        vgg_layer = vgg_test_layers[i]


        dataset = get_dataset("imagenet_val_500")
        eval_(
            layer_str,
            layer_str_to,
            model_str_from="resnet50",
            model_str_to=model_to[i],
            dataset=dataset,
            list_model_str_test=["resnet34", "vgg19_bn", "resnet50"],
            layer_hooks_test=["." + layer_str, vgg_layer, "." + layer_str],
            use_stitching_convolution=use_stitching_convolution[i],
        )



def evaluation_on_afhq():
    layer_strings = (
        ["layer1", "layer2", "layer3", "layer1", "layer2", "layer3"]
        + ["layer1", "layer2", "layer3"]
        + ["layer1", "layer2", "layer3", "layer3"]
    )
    model_to = (
        ["afhqwild.pkl" for i in range(6)]
        + ["imagenet" for i in range(3)]
        + ["BigGAN" for i in range(4)]
    )
    layer_strings_to = (
        [
            "G.synthesis.b128.conv0",
            "G.synthesis.b64.conv0",
            "G.synthesis.b32.conv0",
            "G.synthesis.b128.conv0",
            "G.synthesis.b64.conv0",
            "G.synthesis.b32.conv0",
        ]
        + [
            "G.synthesis.b128.conv0",
            "G.synthesis.b64.conv0",
            "G.synthesis.b32.conv0",
        ]
        + ["G.blocks.3.0", "G.blocks.2.0", "G.blocks.1.0", "G.blocks.2.0"]
    )
    vgg_test_layers = (
        [
            ".features[12]",
            ".features[25]",
            ".features[38]",
            ".features[12]",
            ".features[25]",
            ".features[38]",
        ]
        + [
            ".features[12]",
            ".features[25]",
            ".features[38]",
        ]
        + [
            ".features[12]",
            ".features[25]",
            ".features[38]",
            ".features[38]",
        ]
    )
    use_stitching_convolution = [True, True, True, False, False, False] + [True for i in range(7)]

    for i, layer_str in enumerate(layer_strings):
        torch.manual_seed(42)
        layer_str_to = layer_strings_to[i]
        vgg_layer = vgg_test_layers[i]

        dataset = get_dataset("afhq")
        eval_(
            layer_str,
            layer_str_to,
            model_str_from="resnet50",
            model_str_to=model_to[i],
            dataset=dataset,
            list_model_str_test=["resnet34", "vgg19_bn", "resnet50"],
            layer_hooks_test=["." + layer_str, vgg_layer, "." + layer_str],
            use_stitching_convolution=use_stitching_convolution[i],
        )






if __name__ == "__main__":
    main()
