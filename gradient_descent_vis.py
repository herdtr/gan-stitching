import torchvision
import torch
import torch.nn.functional as F
from lucent.optvis import param
from tqdm import tqdm
import random


# uses the lucent library for regularization

def reconstruct_input_gradient_descent(x, traced_network, num_steps=512, regularization="jitter_only"):
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    with torch.no_grad():
        original_activations = traced_network(x)['layer_output']

    fft = True
    color_decorrelate = True
    if regularization == 'none':
        fft = False
        color_decorrelate = False

    param_f = lambda: param.image(x.shape[-1], fft=fft, decorrelate=color_decorrelate, batch=x.shape[0])
    params, image_f = param_f()

    optimizer = lambda params: torch.optim.Adam(params, lr=5e-2)
    optimizer = optimizer(params)

    upsample = torch.nn.Upsample(size=x.shape[-1], mode="bilinear", align_corners=True)

    for i in tqdm(range(num_steps), ascii=True, disable=True):
        new_val = image_f()
        if regularization == 'jitter_only':
            new_val = normalize(new_val)
            new_val = F.pad(new_val, (2, 2, 2, 2), mode='constant', value=0.5)
            new_val = torch.roll(new_val, (random.randint(-1, 1), random.randint(-1, 1)), (2, 3))

            new_val = upsample(new_val)
        elif regularization == 'none':
            new_val = normalize(new_val)

        reconstructed_activations = traced_network(new_val)["layer_output"]
        loss = F.l1_loss(reconstructed_activations, original_activations)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return image_f().cpu().detach()









