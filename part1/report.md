# Deep Learning Assignment

Data cleaning and preprocessing: 
- All the data given was loaded and concatenated.
- Upon inspection, each image contains 4 digits which are non-overlapping, have 1 channel (grayscale) and have size of 40x168 pixels
- Created a dataloader and split it into training validation and test sets in the ratio 80:10:10 and batch size 32.

## Phase 1: CNN Baselines
- Since a maximum of 4 digits are contained in any image I decided to treat this as a classification problem with 37 classes corresponding to the possible sums. 
- Cross Entropy loss was used for this. 
- Used a basic CNN Architecture with two convolution and pooling layers. Implemented early stopping based on the validation loss.
- Got an accuracy of ~11%
- weights for this model are stored in best_model.pth

- With a similar approach, instead of defining the architecture, a Resnet model was also tested. However, since pretrained resnets have 3 channels (RGB) and use very different data, training the architecture from scratch on the given data seemed to be a better option.
- This performs better with an accuracy of above 50%
- weights for this are stored in best_resnet.pth




