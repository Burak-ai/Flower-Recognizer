Dataset
https://www.kaggle.com/datasets/alxmamaev/flowers-recognition/

Notes

Convolution applies small filters to an image to detect specific patterns 
like edges or textures. The output of each filter is a feature map, showing where 
and how strongly that pattern appears. Multiple filters create multiple feature maps,
capturing different aspects of the image.

In neural networks, a dense layer (also called a fully connected layer) is a layer
 where every neuron is connected to every neuron in the previous layer.
It's mainly used at the end of CNNs to make decisions or predictions

Sequential model is the simplest way to build a neural network in Keras.
It means you stack layers one after another in order

32: Number of filters (feature detectors) — the layer will learn 32 different 
filters. (3, 3): Size of each filter — a 3x3 grid of pixels.

ReLU (Rectified Linear Unit) turns negative values to zero and keeps positive 
values, helping the model learn nonlinear patterns.

128x128 — good balance for many beginner to intermediate projects.
224x224 — standard size used in many famous pretrained models (like VGG, ResNet).

Instead of using 5x5 or 7x7 once, stacking two or three 3x3 layers gives the model
more non-linearity and deeper feature extraction with fewer parameters

MaxPooling2D reduces spatial dimensions (width and height) by half, which:
Makes the model faster and less prone to overfitting.
Helps capture important features by keeping the strongest activations.

Flatten layer converts the 3D feature maps from the last Conv/Pool
layer into a 1D vector so that Dense layers can process them to make predictions.

softmax activation turns outputs into probabilities summing to 1.
The neuron with the highest probability corresponds to the predicted class.
 
Flatten layer converts the 3D feature maps from the last Conv/Pool
layer into a 1D vector so that Dense layers can process them to make predictions.

softmax activation turns outputs into probabilities summing to 1.
The neuron with the highest probability corresponds to the predicted class.
 

Compiling a model is the step where you tell Keras how the model will learn
by choosing an optimizer (how weights update), a loss function
(how errors are measured), and metrics (how performance is tracked)

First use Adam. If the model overfits, behaves oddly,
or you need ultimate control, use SGD with learning rate tuning.

validation_data=validation_generator
This is a separate set of data the model has never seen during training.
After each epoch, the model is tested on validation data to see:
Is it improving?. Is it overfitting?
