# Test
This is a project for mnist data classification.
mnist_net.py script define the neural net work for the training and testing process.
mnist.py contain the code for training and testing during the training process.
Testing accuraccy achieved 97.92%.

test_mnist.py is the script contains the interface class provided by PDF. 

27model.pth is the final pytorch checkpoint which saved after 15 interations.
testing process requires the path of model and then simply call our interface 
eg. out_put = SimpleInfer.infer(image)
Then we can get the prediction of our input image.
(Images which passed to the interface requires a single piece of image with 3 channels.)


------------------------------------------------------------------------------------------------------

#GAN
GAN_image.py is implemented by refer the tutoral online.
Generative Adversarial Networks
discriminator is a network for classification, genorator is a network to take random noises as input 
(in order to randomly generate images) and next of all transform it using a neural net work to produce image.

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
X_gen is passed to Discriminate net and get result of fake image.
Then we calculate loss for gen Net.

Using generator net to perdict images and plot them.


