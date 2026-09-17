# Oxford Pets Breed Lens

Upload a cat or dog image, choose one of the four local ResNet18 checkpoints, and compare the top five predicted breeds.

The checkpoints are trained on 37 Oxford Pet breeds. The app uses each checkpoint's saved image size, normalization values, and class names at inference time.

I have cleared the images and seperate it to its own breed folder. You can use it for further training (I also clean it from kaggle open source dataset >-<).

Please note that I can detect based on 37 breeds only. So, if the your pet's breed is not on the trained list, it will predict the most simlar one.

Have fun learning.
