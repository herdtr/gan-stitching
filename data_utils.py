import torchvision
from dataset import AFHQ_Dataset, ImageNetValDataset_500
from settings import PATH_TO_AFHQ_WILD_VAL, PATH_TO_IMAGENET_VAL


# output of the stylegan2 is at 512x512 and from BigGAN is at 128x128,
# both times not normalized according to what the feature extractor needs as input.
# therefore it needs to be resized and normalized
def get_transform_from_gan_output_to_feature_extractor_input():
    transforms_gan_to_seg = [
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
    return torchvision.transforms.Compose(transforms_gan_to_seg)


def get_dataset(dataset_name):
    transforms_gan_to_seg = get_transform_from_gan_output_to_feature_extractor_input()
    if dataset_name == "afhq":
        transforms = [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
        transforms = torchvision.transforms.Compose(transforms)

        dataset = AFHQ_Dataset(
            transform=transforms, image_folder_path=PATH_TO_AFHQ_WILD_VAL
        )

    elif dataset_name == "imagenet_val_500":
        transforms_crop_resize = [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
        ]
        transforms_crop_resize = torchvision.transforms.Compose(transforms_crop_resize)
        dataset = ImageNetValDataset_500(
            transform=transforms_gan_to_seg,
            transform_crop_resize=transforms_crop_resize,
            image_folder_path=PATH_TO_IMAGENET_VAL,
        )

    return dataset
