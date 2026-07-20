## Downloading Testing Data

SPAI was tested across an evaluation set of 13 generative models and 5 sources of real images. 
To facilitate the reproduction of the evaluation results, the `data` directory contains
a separate CSV file for each of these 18 sources.

As the evaluation set builds upon several pre-existing datasets, the following datasets
should be downloaded and placed under the `data` directory:
- [Synthbuster](https://zenodo.org/records/10066460)
- [ImageNet Test Set](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)
- [Open Images Test Set](https://storage.googleapis.com/openimages/web/download_v7.html)
- [COCO 2017 Test Set](https://cocodataset.org/#download)
- [FODB](https://faui1-files.cs.fau.de/public/mmsec/datasets/fodb/)
- [RAISE-1k](https://loki.disi.unitn.it/RAISE/download.html)

Furthermore, we further facilitate the downloading of Stable Diffusion 3, MidJourney v6.1, 
GigaGAN and Flux images by making them available [here](https://drive.google.com/file/d/1no5T89h97TZvAKNCHKt2PQfKZ1UDtbI4/view?usp=sharing). 
