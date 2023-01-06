

# Importing Essential Libraries

from sklearn import metrics, datasets, svm, model_selection
import numpy as np
import matplotlib.pyplot as plt


# Loading dataset from sklearn built-in dataset 

data = datasets.load_digits()


# Assigning the images and their labels in to the variables

images = data.images
targets = data.target


# We reshape the images to be array of an image containing 64 pixels represents one image as we have 8 by 8 gray scale images so we reshape those size to an arrayy of 64 items each image 

x = np.array(images).reshape(len(images), -1)
y = np.array(targets)


# Splitting the dataset into training set and test set so model can train on training dataset and can be tested on unseen data (test set)

feature_train, feature_test, target_train, target_test = model_selection.train_test_split(x, y, test_size=0.25)


# Visualizing a sample of the dataset so we can have the concept of what kind of images do we have and their labels
# At first we Reshape the sample to be plotted as image using matplotlib
# as we have images of 64 pixels so we reshape it 8 by 8 and plot to the screen with the target label

sample = np.array(feature_train[0]).reshape(8, 8)
plt.imshow(sample, cmap="gray")
plt.title(str(target_train[0]))
plt.show()


# ____________________________ Building The Model ____________________________

model = svm.SVC()
model.fit(feature_train, target_train)
result = model.predict(feature_test)

# Printing model accuracy
print("Model Is Accurate : ", metrics.accuracy_score(target_test, result,), "%")


# Now that the model is trained and tested and reached over 99% accuracy so i would like to test it on one of the test samples and i will plot the real image and label too so to check how the model is working

print("\n____________________________________\n")
sample_for_test = np.array(feature_test[12]).reshape(8, 8)
plt.imshow(sample_for_test, cmap="gray")
plt.title(str(target_test[12]))
plt.show()

model_predection = model.predict([feature_test[12]])
print(" Model Predicted This Number is : ", model_predection[0])

