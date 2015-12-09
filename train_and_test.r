#!/usr/bin/Rscript
# including implementations
source(file = "loader.r")
source(file = "logregression.r")

# loading training data from files
data <- loadMNISTData("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte")
labels <- data$labels
samples <- data$data

# training and evaluating on train dataset
classifier <- train_lr_model(data = samples, labels = labels)
prediction <- test_lr_model(classifier, samples)
cat(sprintf("accuracy on train dataset: %3.1f%%\n", 100 * sum(prediction == labels) / nrow(labels)))

# loading testing data from files
data <- loadMNISTData("mnist/t10k-images-idx3-ubyte", "mnist/t10k-labels-idx1-ubyte")
labels <- data$labels
samples <- data$data

# evaluating on test dataset
prediction <- test_lr_model(classifier, samples)
cat(sprintf("accuracy on test dataset: %3.1f%%\n", 100 * sum(prediction == labels) / nrow(labels)))

#calculate the following error metric for each class obtained on the test data:
#Recall, precision, specificity, F-measure, FDR and ROC for each class separately. Use a package for ROC.
