# NOTE: use the following command to install a package:
#       install.packages(PKG_NAME, lib = PATH)
library("gplots", lib.loc = "./libs/")
library("ROCR", lib.loc = "./libs")

detailed_info <- function(predicted, gt) {
  # The function prints out detailed information about classifiers performance
  for (digit in 0:9) {
    digit_gt <- gt == digit
    digit_prediction <- prediction == digit
    true_positive  <- sum((digit_gt == T) & (digit_prediction == T))
    true_negative  <- sum((digit_gt == F) & (digit_prediction == F))
    false_positive <- sum((digit_gt == F) & (digit_prediction == T))
    false_negative <- sum((digit_gt == T) & (digit_prediction == F))
    gt_positives   <- sum(digit_gt)
    gt_negatives   <- nrow(gt) - gt_positives
    pred_positives <- sum(digit_prediction)
    recall         <- true_positive / gt_positives
    precision      <- true_positive / pred_positives
    specificity    <- true_negative / gt_negatives
    fscore         <- 2 * recall * precision / (recall + precision)
    fdr            <- 1 - precision
    cat(sprintf("Digit \"%d\":\n", digit))
    cat(sprintf("    recall:      %4.1f%%\n", 100 * recall))
    cat(sprintf("    precision:   %4.1f%%\n", 100 * precision))
    cat(sprintf("    specificity: %4.1f%%\n", 100 * specificity))
    cat(sprintf("    f-measure:   %4.1f%%\n", 100 * fscore))
    cat(sprintf("    FDR:         %4.1f%%\n", 100 * fdr))
  }
}

plot_roc_curve <- function(probabilities, gt, output_file) {
  # The function produces ROC curves plots for each binary classifier:
  # `probabilities` -- is a matrix where each row represent the corresponding
  #                    digit classifier predicted probability
  # `gt` -- is ground-truth labels
  # `output_file` -- is a pattern of filenames for plots
  for (digit in 0:9) {
    digit_gt <- (gt == digit) * 1
    roc_prediction_obj <- prediction(probabilities[, digit + 1], digit_gt)
    roc_performance_obj <- performance(roc_prediction_obj, "tpr", "fpr")
    png(filename = sprintf(output_file, digit))
    plot(roc_performance_obj)
  }
}
