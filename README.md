# Test
This is a project for mnist data classification.
mnist_net.py script define the neural net work for the training and test process.
mnist.py contain the code for training and testing during the training process.
testing accuraccy achieved over 97%.

test_mnist.py is the script contains the interface class provided by PDF. 

27model.pth is the pytorch model which saved after 30 interations.
testing process requires the path of model and then simply call our interface 
eg. out_put = SimpleInfer.infer(image)
Then we can get the prediction of our input image.
image which passed to the interface represents a single piece of image with 3 channels.
------------------------------------------------------------------------------------------------------
GAN_image.py is implemented by refer the tutoral online.
Generative Adversarial Networks
discriminator is a network for classification, genorator is a network to take random nnoise as input (in order to randomly generate images) and next of all transform it using a neural net work to produce image.

Real images are passed to Discriminate net 
Here we implement torch.nn.BCEWithLogitsLoss() as the loss function.
We use this loss function with the format as below
loss = binary_cross_entropy(pred, y)
Part 1
noise is passed to generator net and get X_gen result.
X_gen is passed to Discriminate net and get result fake image
X_train is passed to Discriminate net and get the real image
calculate loss for Dis net

Part 2
noise is passed to generator net and get X_gen result.
X_gen is passed to Discriminate net and get result fake image
calculate loss for gen Net.

use generator net to perdict images and plot them.


