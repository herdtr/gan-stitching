import torchvision
import torch
import numpy as np
from tqdm import tqdm
import time

from torchvision.models.feature_extraction import create_feature_extractor

from train import load_model
from gradient_descent_vis import reconstruct_input_gradient_descent
from core import access_activations_forward_hook, cossim_gram_matrices_two_batches

from data_utils import get_dataset


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


def eval_(
    model_str_from,
    list_model_str_test,
    layer_hook_str_list,
    return_layer,
    dataset_name="afhq",
    regularization="jitter_only",
):
    model_from = load_model(model_str_from)

    print("create feature extractor...")
    model_till_layer = create_feature_extractor(
        model_from, return_nodes={return_layer: "layer_output"}
    )

    dataset = get_dataset(dataset_name)

    print("create data loader...")
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=50, shuffle=False, num_workers=8, pin_memory=False
    )

    print("load models...")
    test_models = []
    for i, model_str in enumerate(list_model_str_test):
        test_model = load_model(model_str)
        test_models.append(test_model)

    n_imgs_sampled = 0
    result_metrics = {test_model_str: {} for test_model_str in list_model_str_test}
    for b_data in tqdm(train_loader, ascii=True):
        if n_imgs_sampled == 0:
            start_time = time.time()
        n_imgs_sampled += 1
        x = b_data[0]
        x_transformed = x.to("cuda")

        # print("run reconstruction...")
        x_reconstructed = reconstruct_input_gradient_descent(
            x_transformed,
            model_till_layer,
            num_steps=512,
            regularization=regularization,
        )
        # print("reconstruction shape: {}".format(x_reconstructed.shape))
        input_transfer_transformed = torchvision.transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        )(x_reconstructed).to("cuda")

        for i, model_str in enumerate(list_model_str_test):
            test_model = test_models[i]

            activation_transfer = access_activations_forward_hook(
                [input_transfer_transformed],
                test_model.forward,
                eval("test_model" + layer_hook_str_list[i]),
            ).to("cuda")
            activation_original = access_activations_forward_hook(
                [x.to("cuda")],
                test_model.forward,
                eval("test_model" + layer_hook_str_list[i]),
            ).to("cuda")

            metrics_step = run_metrics(activation_transfer, activation_original)
            for key in metrics_step:
                if not key in result_metrics[model_str].keys():
                    result_metrics[model_str][key] = []
                for value in metrics_step[key]:
                    result_metrics[model_str][key].append(value.cpu())

    print("---------------------------------------------------------------------------")
    print("regularization: {}".format(regularization))
    print(return_layer)
    for key in result_metrics.keys():
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("model string: {}".format(key))
        for metric_key in result_metrics[key]:
            metric_values = result_metrics[key][metric_key]
            mean_value = torch.mean(torch.from_numpy(np.array(metric_values)))
            std_value = torch.std(torch.from_numpy(np.array(metric_values)))
            print(metric_key + " mean: {}".format(mean_value))
            print(metric_key + " standard deviation: {}".format(std_value))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("time taken: {}".format(time.time() - start_time))
    print("---------------------------------------------------------------------------")

    print("done!")



def run_evaluation(dataset_name="afhq"):
    eval_(
        "resnet50",
        ["resnet34", "vgg19_bn", "resnet50"],
        [".layer1", ".features[12]", ".layer1"],
        return_layer="layer1.2.relu_2",
        dataset_name=dataset_name,
        regularization="none",
    )
    eval_(
        "resnet50",
        ["resnet34", "vgg19_bn", "resnet50"],
        [".layer2", ".features[25]", ".layer2"],
        return_layer="layer2.3.relu_2",
        dataset_name=dataset_name,
        regularization="none",
    )
    eval_(
        "resnet50",
        ["resnet34", "vgg19_bn", "resnet50"],
        [".layer3", ".features[38]", ".layer3"],
        return_layer="layer3.5.relu_2",
        dataset_name=dataset_name,
        regularization="none",
    )
    eval_(
        "resnet50",
        ["resnet34", "vgg19_bn", "resnet50"],
        [".layer4", ".features[51]", ".layer4"],
        return_layer="layer4.2.relu_2",
        dataset_name=dataset_name,
        regularization="none",
    )


    eval_(
        "resnet50",
        ["resnet34", "vgg19_bn", "resnet50"],
        [".layer1", ".features[12]", ".layer1"],
        return_layer="layer1.2.relu_2",
        dataset_name=dataset_name,
        regularization="jitter_only",
    )
    eval_(
        "resnet50",
        ["resnet34", "vgg19_bn", "resnet50"],
        [".layer2", ".features[25]", ".layer2"],
        return_layer="layer2.3.relu_2",
        dataset_name=dataset_name,
        regularization="jitter_only",
    )
    eval_(
        "resnet50",
        ["resnet34", "vgg19_bn", "resnet50"],
        [".layer3", ".features[38]", ".layer3"],
        return_layer="layer3.5.relu_2",
        dataset_name=dataset_name,
        regularization="jitter_only",
    )
    eval_(
        "resnet50",
        ["resnet34", "vgg19_bn", "resnet50"],
        [".layer4", ".features[51]", ".layer4"],
        return_layer="layer4.2.relu_2",
        dataset_name=dataset_name,
        regularization="jitter_only",
    )


def measure_reconstruction_time(dataset_name):

    # reconstruction time is measured without computing the metrics
    eval_("resnet50", [], [], return_layer="layer1.2.relu_2",
        dataset_name=dataset_name, regularization="none") #imagenet_val_500
    eval_("resnet50", [], [], return_layer="layer2.3.relu_2",
        dataset_name=dataset_name, regularization="none")
    eval_("resnet50", [], [], return_layer="layer3.5.relu_2",
        dataset_name=dataset_name, regularization="none")
    eval_("resnet50", [], [], return_layer="layer4.2.relu_2",
        dataset_name=dataset_name, regularization="none")


    eval_("resnet50", [], [], return_layer="layer1.2.relu_2",
        dataset_name=dataset_name, regularization="jitter_only") #imagenet_val_500
    eval_("resnet50", [], [], return_layer="layer2.3.relu_2",
        dataset_name=dataset_name, regularization="jitter_only")
    eval_("resnet50", [], [], return_layer="layer3.5.relu_2",
        dataset_name=dataset_name, regularization="jitter_only")
    eval_("resnet50", [], [], return_layer="layer4.2.relu_2",
        dataset_name=dataset_name, regularization="jitter_only")



run_evaluation(dataset_name="afhq")
run_evaluation(dataset_name="imagenet_val_500")


print("------------------------------------------------------")
print("--------------------measure reconstruction time---------------------")
print("------------------------------------------------------")

measure_reconstruction_time("afhq")
measure_reconstruction_time("imagenet_val_500")

