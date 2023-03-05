from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, \
    Embedding, Concatenate

import numpy as np
import random
import os
import cv2

'''
    This is a script to train a cGAN (conditional Generative Adverserial Network), which is a type of machine learning
    model which is used to train a Generator. A Generator can create a sample (in this case a single image), by taking
    as input an array of noise (essentially random data) and a condition (usually the class we want to generate a
    sample from). This allows us to generate images that don't exist yet, based on training samples
    
    In the example below, we use the famous MNIST dataset, which is a collection of handwritten digits between
    0 and 9. The output of our generator therefore is an image with a single digit on it.
    
    NOTE : as soon as you run this script, it will start training. If you can't train due to hardware limitations,
    try changing the self.batch_size value to something lower. If the training times are too long for you, you
    could reduce the self.numImg value. Even at 10000 images, you should get pretty decent results.

    Author :        Martijn Folmer 
    Date created :  05-03-2023
'''


class TrainGeneratorMnist:
    def __init__(self):
        print("initialized the class")

        self.latent_dimension = 100                         # the size of the Noise we use. More complex data requires larger input noise, but this also creates a large model
        self.n_classes = 10                                 # The amount of classes (corresponds to [0..9] in this case)
        self.inputSize = (28, 28, 1)                        # Size of the MNIST images
        self.numImg = 70000                                 # The number of Mnist images we wish to train on (max 70k)
        self.batch_size = 64                                # The batch size per epochs
        self.n_epochs = 100                                 # How many epochs we wish to train
        self.PathToFolderWithResults = 'resultingImg'       # Path to the folder where we want to save our images
        self.GeneratorFileName = 'generatorMNIST'           # where to save our Generator model
        self.DiscriminatorFileName = 'discriminatorMNIST'   # where to save our Discriminator model

        # Create Results folder if it doesn't already exist
        if not os.path.exists(self.PathToFolderWithResults):
            os.mkdir(self.PathToFolderWithResults)

        # Create our generator
        self.generator = self.CreateGeneratorModel(self.latent_dimension, self.n_classes)
        self.generator.summary()
        print("Initialized the generator")

        # Create a discriminator
        self.discriminator = self.CreateDiscriminatorModel(input_shape=self.inputSize, n_classes=self.n_classes)
        self.discriminator.summary()
        print("Initialized the discriminator")

        # load dataset
        self.all_digits, self.all_labels = self.GetMnistData(num_samples=self.numImg)
        print("Loaded Mnist samples")

        # Create the GAN model, which consists of both a generator and discriminator working together
        self.gan_model = self.define_gan(self.generator, self.discriminator)
        print("Defined the gan model")

    def ClearResultsFolder(self, _pathToFolderWithResultingImg):
        """
        Create the folder where we store all of our results. If the folder does not already exist, we create one
        :param _pathToFolderWithResultingImg: The path to the folder.
        """

        # Create the folder if it doesn't already exist
        if not os.path.exists(_pathToFolderWithResultingImg):
            os.mkdir(_pathToFolderWithResultingImg)
            return 0

        # removing all test images
        for fold in [f'{_pathToFolderWithResultingImg}/' + f for f in os.listdir(f'{_pathToFolderWithResultingImg}')]:
            [os.remove(fold + "/" + f) for f in os.listdir(fold)]
        [os.rmdir(f'{_pathToFolderWithResultingImg}/' + fold) for fold in os.listdir(f'{_pathToFolderWithResultingImg}')]



    def GetSampleFakeDiscriminator(self, _generator_model, _batch_size, _latent_dimension, _numclasses):
        """
        This function generates several training samples that we can feed into the Discriminator for training

        :param _generator_model: The generator model which generates the images from an input of labels and Noise
        :param _batch_size: The number of samples we want to generate
        :param _latent_dimension: The size of the noise vector that we feed to the generator
        :param _numclasses: The number of classes we are currently training on.
        :return: Training samples we can feed into discriminator.
        """

        # Create the noise and labels that we use as input for our generator model
        Noise = self.GetLatentSamples(_latent_dimension, _batch_size)
        input_label = np.asarray([np.asarray(random.randint(0, self.n_classes - 1), dtype=np.float32) for _ in range(_batch_size)])

        # Generate output using our generator model
        GeneratedImg = _generator_model.predict((Noise, input_label))

        # The output of the Discriminator model = whether it is a real of fake sample (0=fake, 1 = true)
        output = np.zeros([_batch_size, 1])  # The output of the discriminator are all zero, because they are Fake
        # The input of the Discriminator = [Generated Images, Image labels],
        input = [np.asarray(GeneratedImg, dtype=np.float32), np.asarray(input_label, dtype=np.float32)]
        
        return input, output


    def GetSampleRLDiscriminator(self, _batch_size):
        """
        This function takes several training images from out dataset to feed to the discriminator model
        
        :param _batch_size: The number of images
        :return: The input and output that we can feed to the Discriminator model.
        """
        
        input_img, input_label = [], []
        for _ in range(_batch_size):
            ir = random.randint(0, self.numImg-1)    
            img = self.all_digits[ir]                # get random image from dataset
            img = (img[:, :]-128.0)/255.0            # normalize images between -1 and 1
            input_img.append(img)
            input_label.append(self.all_labels[ir])  # append the correct label
        input_img = np.asarray(input_img, dtype=np.float32)
        input_img = np.reshape(input_img, (input_img.shape[0], input_img.shape[1], input_img.shape[2], 1))

        output = np.ones([_batch_size, 1])            # The output of the discriminator are all one, because RL img
        input = [input_img, np.asarray(input_label, dtype=np.float32)]
        return input, output

    def GetLatentSamples(self, _latent_dimension, _batch_size):
         """
         This function generates "Noise", which is what we feed to the generator (along with a label) to generate
         a sample.
         :param _latent_dimension: The size of the noise that we want to generate
         :param _batch_size: How many samples we want to generate
         :return: An array of size [batch_size, latent_dimension], with number between -1 and 1.
         """
         return np.random.normal(loc=0.0, scale=1.0, size=(_batch_size, _latent_dimension))

    def GetMnistData(self, num_samples):
        """
        This function downloads the Mnist data that we wish to train on
        :param num_samples: On how many samples we wish to train. Maximum is 70000
        :return: all_img, which contains the [28x28x1] Mnist images, and all_labels, which contains the label for each
                        of the img in all_img (a number of 0-9, corresponding with which digit is represented)
        """

        # Load the data and merge the train and test data
        (x_train, y_train), (x_test, y_test) = load_data()
        all_img = np.concatenate([x_train, x_test])
        all_labels = np.concatenate([y_train, y_test])

        # Limit the amount of samples we want to train on
        all_img = all_img[:num_samples]
        all_labels = all_labels[:num_samples]

        return all_img, all_labels

    def CreateGeneratorModel(self,latent_dim, n_classes):

        """
        The generator takes "noise" and a label (between 0 and 9) as input, and upsamples it into an output image

        :param latent_dim: The dimensions of the latent noise that we use as an input
        :param n_classes: The amount of classes we want to train on
        :return: an [28x28x1] image which should be an image of the label we passed into it.
        """

        # First Channel input : the label of the class we wish to generate for
        LabelInputFirst = Input(shape=(1,))
        # embedding for categorical input, embedding layers allow us to transform labels into vectors of fixed length
        LabelInput = Embedding(n_classes, 50)(LabelInputFirst)

        # make it so we have a 7x7xN image which we can concatenate with noise. (we use N=1, but can be any positive integer)
        n_nodes = 7 * 7
        LabelInput = Dense(n_nodes)(LabelInput)
        LabelInput = Reshape((7, 7, 1))(LabelInput)

        # Second Channel input : the noise we generate the image from
        NoiseInputFirst = Input(shape=(latent_dim,))
        n_nodes = 128 * 7 * 7
        NoiseInput = Dense(n_nodes)(NoiseInputFirst)
        NoiseInput = LeakyReLU(alpha=0.2)(NoiseInput)
        NoiseInput = Reshape((7, 7, 128))(NoiseInput)

        # merge image gen and label input
        mergedInputs = Concatenate()([NoiseInput, LabelInput])   # concatenates the 7x7 images from noise and claas id

        # upsample to 14x14
        upsample1 = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(mergedInputs)
        upsample1 = LeakyReLU(alpha=0.2)(upsample1)

        # upsample to 28x28
        upsample2 = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(upsample1)
        upsample2 = LeakyReLU(alpha=0.2)(upsample2)

        # output : Generate the output layer (tanh activation makes it between -1 and 1)
        out_layer = Conv2D(1, (7, 7), activation='tanh', padding='same')(upsample2)

        # define model : input = noise  and class id, output=generated image
        model = Model([NoiseInputFirst, LabelInputFirst], out_layer)
        return model

    def CreateDiscriminatorModel(self, input_shape=(28, 28, 1), n_classes=10):
        """
        This function creates our Discriminator model, which determines whether a given combination of image and
        label is Real or False. The idea is that we train this model to get better at distinguishing fakes, which
        forces the Generator model to become better in order to fool the Discriminator

        :param input_shape: The shape of the image we use as an input
        :param n_classes: The number of classes we are training on
        :return: The Discriminator model
        """

        # Label Input
        LabelInputFirst = Input(shape=(1,))

        # embedding for categorical input, embedding layers allow us to transform labels into vectors of fixed length
        LabelInput = Embedding(n_classes, 50)(LabelInputFirst)
        n_nodes = input_shape[0] * input_shape[1]
        LabelInput = Dense(n_nodes)(LabelInput)
        # Rehsape to have the same size as the size of the input of the images
        LabelInput = Reshape((input_shape[0], input_shape[1], 1))(LabelInput)

        # Image input
        ImageInput = Input(shape=input_shape)

        # Merge both input image and embedding image (so size after merge is [28x28x2]
        # concat label as a channel
        merge = Concatenate()([ImageInput, LabelInput])

        # Downsample from our input towards the output (which is [0,1])
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Flatten()(x)  # Create Feature map
        x = Dropout(0.2)(x)

        # Turn to output (sigmoid, because it is either 0, for false or 1, for real)
        out_layer = Dense(1, activation='sigmoid')(x)

        # define model
        model = Model([ImageInput, LabelInputFirst], out_layer)

        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def define_gan(self, generator, discriminator):

        """
        This is the GAN model, which we train to improve the generator.

        :param generator: The generator model, that outputs images based on noise + labels
        :param discriminator: The discriminator model, which tries to distinguish between real images and those
            generated by the generator model
        :return: the GAN model.
        """

        # We don't train the discriminator during the GAN training, only the generator.
        discriminator.trainable = False
        # get noise and label inputs and image output of the generator model
        gen_noise, gen_label = generator.input
        gen_output = generator.output

        # The output of the Gan is the output of the discriminator
        gan_output = discriminator([gen_output, gen_label])

        # define gan model as taking noise and label and outputting a classification (of 0 for false and 1 for real)
        model = Model([gen_noise, gen_label], gan_output)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def train(self, generator, discriminator, gan_model, dataset_size, latent_dim, numclasses, n_epochs=1, n_batch=128):
        """
        This function trains the GAN, and in turn, the discriminator and generator models. Periodically, it will
        generate images from our generator model and save them in the folder defined by self.PathToFolderWithResults
        
        :param generator: The generator model, which generates images based on noise and label input
        :param discriminator: The discriminator model, which classifies images as real or fake
        :param gan_model: The gan model, consisting of both the generator and discriminator working together
        :param dataset_size: The size of the MNIST data we have loaded
        :param latent_dim: The size of the Noise we use in the generator
        :param numclasses: The number of classes we are training on
        :param n_epochs: The amount of epochs we wish to train for
        :param n_batch: The size of a single training batch

        """

        bat_per_epo = int(dataset_size / n_batch)  # number of steps per epoch
        # for the batch that we train the discriminator on, half are real, and half are fake
        half_batch = int(n_batch / 2)

        # Each epoch consists of training the discriminator on a batch, then the GAN on a batch, until we have looped
        # through all data
        for i in range(n_epochs):
            for j in range(bat_per_epo):

                # Real life samples
                [X_real, labels_real], y_real = self.GetSampleRLDiscriminator(half_batch)
                # Fake samples
                [X_fake, labels_fake], y_fake = self.GetSampleFakeDiscriminator(generator, half_batch, latent_dim, numclasses)

                X = np.concatenate([X_real, X_fake])
                labels = np.concatenate([labels_real, labels_fake])
                y = np.concatenate([y_real, y_fake])

                # Shuffle the real and fake samples together
                indices = np.random.permutation(X.shape[0])
                X, labels, y = X[indices], labels[indices], y[indices]

                # Train the Discriminator
                discriminator_loss, _ = discriminator.train_on_batch([X, labels], y)

                # prepare points in latent space as input for the generator
                Noise = self.GetLatentSamples(self.latent_dimension, self.batch_size)
                labels_input = np.asarray([np.asarray(random.randint(0, self.n_classes-1)) for _ in range(self.batch_size)])
                y_gan = np.ones((n_batch, 1))  # Train the generator in such a way that the discriminator outputs 1

                # Train GAN, which trains the generator
                generator_loss = gan_model.train_on_batch([Noise, labels_input], y_gan)
                # summarize loss on this batch
                print('>%d, %d/%d, discriminator_loss=%.3f, generator_loss=%.3f' %
                      (i + 1, j + 1, bat_per_epo, discriminator_loss, generator_loss))

            # Predictions every 10 epochs
            if (i+1) % 1 == 0:
                foldCur = f'{self.PathToFolderWithResults}/epoch_{i+1}'
                if not os.path.exists(foldCur):
                    os.mkdir(foldCur)

                # predict and show images
                all_index = [np.asarray([i]) for i in range(0, self.n_classes)]

                kn = 0
                for input_label in all_index:
                    Noise = self.GetLatentSamples(self.latent_dimension, 1)
                    GeneratedImg = self.generator.predict((Noise, input_label))
                    GeneratedImg = (GeneratedImg[:, :] * 128.0) + 128.0
                    GeneratedImg = np.asarray(GeneratedImg, dtype=np.uint8)
                    GeneratedImg = np.reshape(GeneratedImg, (28, 28, 1))

                    cv2.imwrite(f'{foldCur}/img_{kn}_label_{input_label[0]}.png', GeneratedImg)
                    kn += 1

    def StartTraining(self):
        # Actually train the GAN
        print("Commence the training!")
        self.train(self.generator, self.discriminator, self.gan_model, self.all_digits.shape[0], self.latent_dimension,
                   self.n_classes, self.n_epochs, self.batch_size)
        # Save our models
        self.SaveModel(self.generator, self.GeneratorFileName)
        self.SaveModel(self.discriminator, self.DiscriminatorFileName)

    def SaveModel(self, _model, _folderPath):
        # create the folder that we save the model in if it doesn't already exist
        if not os.path.exists(_folderPath):
            os.mkdir(_folderPath)
        # actually save the model
        _model.save(_folderPath)


# Initialize the class
if __name__ == "__main__":
    TGM = TrainGeneratorMnist()
    TGM.StartTraining()   # This is where we start training the GAN

