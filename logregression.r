train_lr_model <- function(data, labels) {
  # labels are presenting digit classes from 0 to 9:
  # will prepare 10 binary classificators for each digit
  target_error_diff <- 0.1
  regularization_rate <- 0.4
  initial_learning_rate <- 4

  # NOTE: there is no need to normalize the data because it has the same nature.
  # adding constant feature row with 1 to emulate shift of weighted summ
  data <- cbind(1, data)
  samples_count <- nrow(data)
  features_count <- ncol(data)

  # output matrix with linear classifiers coeffs (per column)
  classifier <- matrix(0, nrow = features_count, ncol = 10)

  # optimizing classifiers coeffs
  for (digit_idx in 1:10) {
    cat(sprintf("\n-- training binary classifier for %d digit --\n", digit_idx - 1))
    # initializing the classifier weight with zeros
    weights <- matrix(0, nrow = features_count, ncol = 1)
    # preparing labels for binary classification: 1 for match, 0 for other digit
    gt <- (labels == (digit_idx - 1)) * 1
    prev_error <- Inf
    learning_rate <- initial_learning_rate
    iter_counter <- 1
    while (TRUE) {
      pred <- sigmoid(data %*% weights)
      error <- sum(-gt * log(pred + 1e-12) - (1 - gt) * log(1 + 1e-12 - pred)) / samples_count +
               regularization_rate * sum(weights ^ 2)
      false_negative <- sum((gt == 1) & (pred < 0.5 + 1e-6))
      false_positive <- sum((gt == 0) & (pred > 0.5 - 1e-6))
      cat(sprintf("%03d> raw error: %10.9f [positive class: %5d/%d, negative class: %5d/%d]\n",
                  iter_counter, error, false_negative, sum(gt), false_positive, sum(1 - gt)))
      if (abs(error - prev_error) < target_error_diff) {
        classifier[, digit_idx] <- weights
        break
      }
      prev_error <- error
      grad <- (t(data) %*% (pred - gt)) / samples_count + 2 * regularization_rate * weights
      weights <- weights - learning_rate * grad
      learning_rate <- 0.9 * learning_rate
      iter_counter <- iter_counter + 1
    }
  }
  return(classifier)
}

test_lr_model <- function(classifier, data) {
  # return predicted labels column-matrix
  data <- cbind(1, data)
  predicted_probabilities <- sigmoid(data %*% classifier)
  predicted_labels <- matrix(0, nrow = nrow(data), ncol = 1)
  for (sample_idx in 1:nrow(predicted_probabilities)) {
    predicted_labels[sample_idx] <- which.max(predicted_probabilities[sample_idx,])
  }
  return(predicted_labels - 1)
}

sigmoid <- function(x) {
  return(1.0 / (1.0 + exp(-x)))
}
