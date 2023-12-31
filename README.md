# BROAD SuperUROP (Piximi: Browser-based inference engine for biological image segmentation)

Medical imaging tools serve a critical purpose not only in helping clinical physiologists diagnose illnesses but also supporting biologists to further their research. Despite the data generated by these technologies being immense, which is good for training machine learning models, they differ significantly from natural images in terms of resolution, sharpness, noise levels, color diversity, and deformations. However, biological image segmentation models have been able to push the needle in terms of accuracy, with some model architectures(mostly variants of U-Net) achieving highs of over 97%. 

Despite these accuracy improvements, biologists find it difficult to interact with these tools because of the technical knowledge needed to use them. The figure below from a BioImage analysis survey, shows a preference for simple intuitive interfaces.
This project aims to bring bio image segmentation models to the browser without sacrificing accuracy with a smooth and simple user experience – this implies developing an aggressive optimization focus for segmentation models.

## Preliminary methods
### Fine-tuning segmentation models(in the U-Net family) with pre-trained backbones
Using the Broad Bioimage Benchmark Collection(BBBC) and the Cellpose datasets, I aim to fine-tune models that have already been made efficient using methods such as pruning, quantization, distillation etc. Some models that I intend to fine-tune include the Efficient-SAM family of lightweight models and training general segmentation models that might have a backbone model pretrained on a popular general dataset e.g. ImageNet; such models would include the vanilla U-Net model, Squeeze-UNet, SSD, R-CNN, Mask-RCNN, and its faster variants, and FCN.
I hope to explore the above experiments while comparing performance and accuracy.

### Implementing and training U-Net family segmentation models from scratch then making them efficient
A key part of making models smaller is making them more sparse by pruning, doing quantization, or running distillation. These are direct alterations to the model without changing its architecture. I hope to make these alterations in addition to implementation-specific technical choices such as converting the model to an onnxruntime format(ORT).
I also hope to tweak the architecture of the foundational segmentation models to observe how the tweaks affect accuracy and efficiency.

### Fine-tuning vision-transformer(ViT)-based segmentation models(SAM family) & making them more efficient
Witnessing the success of other (large) transformer based models being fit in the browser(case in point [web llm](https://webllm.mlc.ai/)), I claim that there has to be a way of pulling the same feat with vision-transformer models with decent(or even impressive) performance. Using typical efficiency techniques and more novel ones like weight-sharing, I hope to get a variant of SAM to work well for our use case.

### Dynamically using the GPU when available
Chrome provides an API to the GPU called WebGPU. This would help us to dynamically decide to whether to perform inference on the CPU or otherwise. The GPU will reward us with performance gains that the CPU can’t grant, and this dynamic feature will allow all classes of client devices to work with the Piximi software.

## Conclusion
Using an ensemble of the methods mentioned above, including others such as WebAssembly, I hope these preliminary experiments will take me a step closer to succeeding in my research goal.
