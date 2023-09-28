from measure_evaluation_metrics_gan import eval_

def main():
    models_from = ["resnet50" for i in range(10)]
    models_to = ["BigGAN" for i in range(3)] + ["afhqwild.pkl" for i in range(7)]
    layers_from = ["layer1", "layer2", "layer3"] + [
        "layer1",
        "layer2",
        "layer3",
        "layer4",
    ] + ["layer1", "layer2", "layer3"]
    layers_to = ["G.blocks.3.0", "G.blocks.2.0", "G.blocks.1.0"] + [
        "G.synthesis.b128.conv0",
        "G.synthesis.b64.conv0",
        "G.synthesis.b32.conv0",
        "G.synthesis.b16.conv0",
    ] + ["G.synthesis.b128.conv0", "G.synthesis.b64.conv0", "G.synthesis.b32.conv0"]
    use_stitching_convolution = [True for i in range(7)] + [False for i in range(3)]

    runtimes = []
    for j, model_from in enumerate(models_from):
        model_to = models_to[j]
        layer_from = layers_from[j]
        layer_to = layers_to[j]
        runtime = eval_(
            layer_from,
            layer_to,
            model_str_from=model_from,
            model_str_to=model_to,
            dataset_name="imagenet_val_500",
            list_model_str_test=[],
            layer_hooks_test=[],
            first_ckpt_only=True,
            use_stitching_convolution=use_stitching_convolution[j]
        )

        runtimes.append([runtime, model_to, layer_from, layer_to, use_stitching_convolution[j]])

    for run_data in runtimes:
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("time taken: {}".format(run_data[0]))
        print("model to: {}".format(run_data[1]))
        print("layer from: {}".format(run_data[2]))
        print("layer to: {}".format(run_data[3]))
        print("use stitching convolution: {}".format(run_data[4]))
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


if __name__ == "__main__":
    main()

