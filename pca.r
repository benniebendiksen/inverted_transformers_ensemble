# PCA Analysis and Export for Cryptocurrency Time Series with Proper Train/Val/Test Split
library(readr)
library(tidyverse)
library(stats)

# Set the file path
# file_path <- "/Users/bendiksen/Desktop/iTransformer/dataset/logits/btcusdc_12h_historical.csv"
file_path <- "/Users/bendiksen/Desktop/inverted_transformers_ensemble/binance_us_historical_data/btcusdc_12h_historical.csv"


#===============================================================================
# Data Loading and Preprocessing Functions
#===============================================================================

# Function to read and prepare the data
load_data <- function(file_path) {
  # Read the CSV file
  df <- read_csv(file_path)
  
  # Convert timestamp to proper datetime if needed
  if("timestamp" %in% colnames(df)) {
    df$timestamp <- as.POSIXct(df$timestamp)
  }
  
  # Check for infinite values and replace with NA
  df <- df %>% mutate_all(~ifelse(is.infinite(.), NA, .))
  
  return(df)
}

# Function to split data chronologically with proper train/val/test splits
split_data_chronologically <- function(df, train_ratio = 0.85, valid_ratio = 0.10, test_ratio = 0.05) {
  # Ensure data is ordered chronologically
  if("timestamp" %in% colnames(df)) {
    df <- df %>% arrange(timestamp)
  }
  
  # Calculate split indices
  n <- nrow(df)
  n_train <- floor(n * train_ratio)
  n_valid <- floor(n * valid_ratio)
  n_test <- n - n_train - n_valid  # Remainder goes to test
  
  # Create the splits
  train_idx <- 1:n_train
  valid_idx <- (n_train + 1):(n_train + n_valid)
  test_idx <- (n_train + n_valid + 1):n
  
  # Create split indicators
  split_indicators <- rep(NA, n)
  split_indicators[train_idx] <- "train"
  split_indicators[valid_idx] <- "val"
  split_indicators[test_idx] <- "test"
  
  # Add split column to the dataframe
  df$split <- split_indicators
  
  cat("Data split chronologically with distinct validation and test sets:\n")
  cat("  Training set:   ", length(train_idx), "rows (", train_ratio*100, "%)\n")
  cat("  Validation set: ", length(valid_idx), "rows (", valid_ratio*100, "%)\n")
  cat("  Test set:       ", length(test_idx), "rows (", test_ratio*100, "%)\n")
  
  return(list(
    data = df,
    train_idx = train_idx,
    valid_idx = valid_idx,
    test_idx = test_idx
  ))
}

# Function to handle NA values - fit on training, apply to all
handle_missing_values <- function(df, train_idx) {
  # Calculate medians from training data only
  train_medians <- df[train_idx, ] %>% 
    select_if(is.numeric) %>% 
    summarise(across(everything(), ~median(., na.rm = TRUE)))
  
  # Apply medians to all data
  for(col in names(train_medians)) {
    if(col %in% colnames(df)) {
      df[[col]] <- ifelse(is.na(df[[col]]), train_medians[[col]], df[[col]])
    }
  }
  
  return(df)
}

# Function to identify and remove highly correlated features - using only training data
remove_highly_correlated <- function(df, train_idx, threshold = 0.95) {
  # Calculate the correlation matrix using only training data
  train_data <- df[train_idx, ] %>% select_if(is.numeric)
  cor_matrix <- cor(train_data, use = "complete.obs")
  
  # Find highly correlated pairs
  high_cor <- which(abs(cor_matrix) > threshold & abs(cor_matrix) < 1, arr.ind = TRUE)
  
  # Create a dataframe of highly correlated pairs
  cor_pairs <- data.frame(
    row = rownames(cor_matrix)[high_cor[, 1]],
    col = colnames(cor_matrix)[high_cor[, 2]],
    cor = cor_matrix[high_cor]
  )
  
  # Sort by correlation
  cor_pairs <- cor_pairs %>% arrange(desc(abs(cor)))
  
  # Print top correlated pairs
  cat("Top highly correlated pairs (based on training data only):\n")
  print(head(cor_pairs, 10))
  
  # Identify columns to remove (simplified approach)
  cols_to_remove <- c()
  
  for (i in 1:nrow(cor_pairs)) {
    pair <- cor_pairs[i, ]
    # If neither column has been removed yet
    if (!pair$row %in% cols_to_remove && !pair$col %in% cols_to_remove) {
      # Remove the second column of the pair
      cols_to_remove <- c(cols_to_remove, as.character(pair$col))
    }
  }
  
  cat("\nRemoving", length(cols_to_remove), "highly correlated features (identified from training data only)\n")
  
  # Return data without highly correlated columns (applied to all data)
  return(list(
    data = df %>% select(-one_of(cols_to_remove)),
    removed = cols_to_remove
  ))
}

# Function to identify and remove zero variance columns - using only training data
remove_zero_variance <- function(df, train_idx) {
  # Calculate variance for all numeric columns in training data only
  train_data <- df[train_idx, ] %>% select_if(is.numeric)
  variances <- apply(train_data, 2, var)
  
  # Identify columns with near-zero variance
  zero_var_cols <- names(which(variances < 1e-10))
  
  if(length(zero_var_cols) > 0) {
    cat("\nRemoving", length(zero_var_cols), "columns with zero or near-zero variance in training data:\n")
    print(zero_var_cols)
    
    # Return data without zero variance columns (applied to all data)
    return(list(
      data = df %>% select(-one_of(zero_var_cols)),
      removed = zero_var_cols
    ))
  } else {
    cat("\nNo zero variance columns found in training data.\n")
    return(list(
      data = df,
      removed = character(0)
    ))
  }
}

#===============================================================================
# PCA Analysis Function - Modified to Respect Train/Val/Test Split
#===============================================================================

# Preprocess and prepare data for PCA - now respecting data splits
preprocess_for_pca <- function(df, train_idx) {
  # Extract and save non-numeric columns we want to keep
  preserved_cols <- list()
  preserved_vars <- c("timestamp", "close", "split")
  
  for(var in preserved_vars) {
    if(var %in% colnames(df)) {
      preserved_cols[[var]] <- df[[var]]
      df <- df %>% select(-all_of(var))
    }
  }
  
  # Handle missing values - fit on training, apply to all
  df <- handle_missing_values(df, train_idx)
  
  # Remove highly correlated features - based on training data only
  cat("\n=== REMOVING HIGHLY CORRELATED FEATURES (BASED ON TRAINING DATA) ===\n")
  reduced <- remove_highly_correlated(df, train_idx)
  df_reduced <- reduced$data
  removed_correlated <- reduced$removed
  cat("Dataset reduced to", ncol(df_reduced), "columns\n")
  
  # Remove zero variance features - based on training data only
  cat("\n=== REMOVING ZERO VARIANCE FEATURES (BASED ON TRAINING DATA) ===\n")
  reduced_var <- remove_zero_variance(df_reduced, train_idx)
  df_final <- reduced_var$data
  removed_zero_var <- reduced_var$removed
  cat("Final dataset has", ncol(df_final), "columns\n")
  
  # Return all necessary information
  return(list(
    processed_data = df_final,
    preserved_cols = preserved_cols,
    removed = list(
      correlated = removed_correlated,
      zero_var = removed_zero_var
    )
  ))
}

# Function to perform PCA - fit on training data only, then transform all data
analyze_pca_components <- function(processed_data, train_idx) {
  # Select only numeric columns
  numeric_df <- processed_data %>% select_if(is.numeric)
  
  # Compute scaling parameters from training data only
  train_center <- colMeans(numeric_df[train_idx, ], na.rm = TRUE)
  train_scale <- apply(numeric_df[train_idx, ], 2, sd, na.rm = TRUE)
  
  # Handle any zero scale values
  zero_scale_cols <- names(which(train_scale == 0))
  if(length(zero_scale_cols) > 0) {
    cat("\nWarning:", length(zero_scale_cols), "columns have zero standard deviation in training data.\n")
    cat("These columns will be removed:", paste(zero_scale_cols, collapse=", "), "\n")
    
    # Remove problematic columns
    numeric_df <- numeric_df %>% select(-all_of(zero_scale_cols))
    
    # Recalculate scaling parameters
    train_center <- train_center[!names(train_center) %in% zero_scale_cols]
    train_scale <- train_scale[!names(train_scale) %in% zero_scale_cols]
  }
  
  # Standardize all data using training parameters
  scaled_data <- scale(numeric_df, center = train_center, scale = train_scale)
  
  # Store scaling parameters for later use
  scaling_params <- list(
    center = train_center,
    scale = train_scale
  )
  
  # Fit PCA on training data only
  pca_result <- prcomp(scaled_data[train_idx, ], center = FALSE, scale. = FALSE)
  
  # Calculate variance explained (based on training data)
  variance <- pca_result$sdev^2
  prop_variance <- variance / sum(variance)
  cum_variance <- cumsum(prop_variance)
  
  # Create summary data frame for all components
  variance_df <- data.frame(
    PC = 1:length(prop_variance),
    Variance = prop_variance,
    Cumulative = cum_variance
  )
  
  # Print first 15 PCs
  cat("\nExplained variance by principal components (based on training data):\n")
  print(head(variance_df, 15))
  
  # Find key variance thresholds
  if(any(cum_variance >= 0.7)) {
    n_70 <- which(cum_variance >= 0.7)[1]
    cat("\n", n_70, "components needed to explain 70% of variance\n")
  }
  
  if(any(cum_variance >= 0.8)) {
    n_80 <- which(cum_variance >= 0.8)[1]
    cat("\n", n_80, "components needed to explain 80% of variance\n")
  }
  
  if(any(cum_variance >= 0.9)) {
    n_90 <- which(cum_variance >= 0.9)[1]
    cat("\n", n_90, "components needed to explain 90% of variance\n")
  }
  
  if(any(cum_variance >= 0.95)) {
    n_95 <- which(cum_variance >= 0.95)[1]
    cat("\n", n_95, "components needed to explain 95% of variance\n")
  }
  
  # Create scree plot
  pdf("pca_scree_plot.pdf", width=10, height=6)
  
  # Plot individual variance
  par(mfrow=c(1,2))
  plot(variance_df$PC, variance_df$Variance, type="b", 
       main="Scree Plot - Individual Variance", 
       xlab="Principal Component", ylab="Proportion of Variance Explained",
       col="steelblue", pch=19)
  
  # Add elbow point markers
  abline(h=0.01, col="red", lty=2)  # Often used as a cutoff
  abline(h=mean(prop_variance), col="darkgreen", lty=2)  # Mean variance
  
  # Plot cumulative variance
  plot(variance_df$PC, variance_df$Cumulative, type="b", 
       main="Cumulative Variance Explained", 
       xlab="Number of Principal Components", 
       ylab="Cumulative Proportion of Variance Explained",
       col="steelblue", pch=19)
  
  # Add threshold lines
  abline(h=0.7, col="green", lty=2)
  abline(h=0.8, col="blue", lty=2)
  abline(h=0.9, col="red", lty=2)
  abline(h=0.95, col="purple", lty=2)
  
  # Add legend
  legend("bottomright", 
         legend=c("70%", "80%", "90%", "95%"),
         col=c("green", "blue", "red", "purple"), 
         lty=2, cex=0.8)
  
  dev.off()
  
  # Create table for rule-of-thumb component selection methods
  cat("\n=== COMPONENT SELECTION RECOMMENDATIONS ===\n")
  
  # Kaiser rule (eigenvalues > 1)
  kaiser_n <- sum(variance > 1)
  cat("Kaiser rule (eigenvalues > 1):", kaiser_n, "components\n")
  
  # Variance explained thresholds
  cat("70% variance threshold:", if(exists("n_70")) n_70 else "N/A", "components\n")
  cat("80% variance threshold:", if(exists("n_80")) n_80 else "N/A", "components\n")
  cat("90% variance threshold:", if(exists("n_90")) n_90 else "N/A", "components\n")
  
  # Proportion of max explained variance (> 5% of max variance)
  var_cutoff <- 0.05 * max(prop_variance)
  prop_max_n <- sum(prop_variance > var_cutoff)
  cat("Components explaining >5% of max variance:", prop_max_n, "components\n")
  
  # Average variance criterion
  avg_var_n <- sum(prop_variance > mean(prop_variance))
  cat("Average variance criterion:", avg_var_n, "components\n")
  
  # Now transform all data using the PCA fitted on training data
  all_transformed <- predict(pca_result, scaled_data)
  
  # Return variance data and PCA results for further analysis
  return(list(
    pca_result = pca_result,
    all_transformed = all_transformed,
    scaling_params = scaling_params,
    variance_df = variance_df,
    selected_components = list(
      kaiser = kaiser_n,
      var_70 = if(exists("n_70")) n_70 else NA,
      var_80 = if(exists("n_80")) n_80 else NA,
      var_90 = if(exists("n_90")) n_90 else NA,
      prop_max = prop_max_n,
      avg_var = avg_var_n
    ),
    removed_cols = if(exists("zero_scale_cols")) zero_scale_cols else character(0)
  ))
}

#===============================================================================
# Main Analysis Execution Function - With Proper Train/Val/Test Splits
#===============================================================================

# Main execution - perform full PCA analysis with proper train/val/test splits
run_pca_analysis <- function(file_path, train_ratio = 0.85, valid_ratio = 0.10, test_ratio = 0.05) {
  # Load data
  cat("Loading data...\n")
  df <- load_data(file_path)
  cat("Dataset loaded:", nrow(df), "rows,", ncol(df), "columns\n")
  
  # Split data chronologically with proper train/val/test splits
  cat("\n=== SPLITTING DATA CHRONOLOGICALLY ===\n")
  split_data <- split_data_chronologically(df, train_ratio, valid_ratio, test_ratio)
  df_split <- split_data$data
  train_idx <- split_data$train_idx
  valid_idx <- split_data$valid_idx
  test_idx <- split_data$test_idx
  
  # Preprocess data for PCA - using only training data for fitting
  cat("\n=== PREPROCESSING DATA (USING TRAIN DATA FOR PARAMETERS) ===\n")
  preprocessed <- preprocess_for_pca(df_split, train_idx)
  
  # Run PCA analysis - fit on training data, transform all
  cat("\n=== PCA COMPONENT ANALYSIS (FIT ON TRAIN DATA ONLY) ===\n")
  pca_results <- analyze_pca_components(preprocessed$processed_data, train_idx)
  
  # Store all necessary information for later export
  full_results <- list(
    split_data = df_split,
    train_idx = train_idx,
    valid_idx = valid_idx,
    test_idx = test_idx,
    processed_data = preprocessed$processed_data,
    preserved_cols = preprocessed$preserved_cols,
    removed_features = preprocessed$removed,
    pca_results = pca_results
  )
  
  # Save the full results for later use (optional)
  saveRDS(full_results, "pca_full_results_proper_splits.rds")
  
  return(full_results)
}

#===============================================================================
# Export Functions - With Proper Train/Val/Test Splits
#===============================================================================

# Export PCA-transformed dataset with proper train/val/test splits
export_pca_components <- function(results, n_components, output_file = NULL) {
  # Extract necessary data from results
  all_transformed <- results$pca_results$all_transformed
  preserved_cols <- results$preserved_cols
  
  # Validate n_components
  max_components <- ncol(all_transformed)
  if (n_components > max_components) {
    warning(paste("n_components exceeds available principal components (", max_components, "). Using max available."))
    n_components <- max_components
  }
  
  # Select desired number of components
  pc_selected <- as.data.frame(all_transformed[, 1:n_components])
  
  # Add meaningful column names
  colnames(pc_selected) <- paste0("PC", 1:n_components)
  
  # Prepare final dataset with preserved columns
  final_data <- pc_selected
  
  # Add timestamp as first column if it exists
  if (!is.null(preserved_cols$timestamp)) {
    cat("Adding timestamp as first column\n")
    final_data <- cbind(date = preserved_cols$timestamp, final_data)
    colnames(final_data)[1] <- "date"  # Rename to match expected format
  }
  
  # Add close price as last column if it exists
  if (!is.null(preserved_cols$close)) {
    cat("Adding close price as last column\n")
    final_data$close <- preserved_cols$close
  }
  
  # Add split column to preserve train/val/test data split information
  if (!is.null(preserved_cols$split)) {
    cat("Adding train/validation/test split information\n")
    final_data$split <- preserved_cols$split
  }
  
  # Define export path
  if (is.null(output_file)) {
    output_file <- paste0("btcusdc_pca_components_", n_components, ".csv")
  }
  export_path <- file.path(dirname(file_path), output_file)
  
  # Export the dataset
  write_csv(final_data, export_path)
  cat("Successfully exported", n_components, "principal components to:", export_path, "\n")
  
  # Print summary counts by split
  if (!is.null(preserved_cols$split)) {
    split_counts <- table(final_data$split)
    cat("\nExported data summary by split:\n")
    print(split_counts)
  }
  
  return(final_data)
}

# Export multiple versions with different component counts
export_multiple_versions <- function(results, component_counts, base_filename = "btcusdc_pca_components") {
  exported_datasets <- list()
  
  for (n in component_counts) {
    output_file <- paste0(base_filename, "_", n, ".csv")
    exported_datasets[[as.character(n)]] <- export_pca_components(results, n, output_file)
  }
  
  return(exported_datasets)
}

#===============================================================================
# Script Execution with Proper Train/Val/Test Splits
#===============================================================================

# Run the full analysis with 85% training, 10% validation, 5% test
results <- run_pca_analysis(file_path, train_ratio = 0.90, valid_ratio = 0.05, test_ratio = 0.05)

# Export dataset with specified number of components
# Adjust the number of components based on your PCA results
# For example, to capture 95% of variance:
export_pca_components(results, n_components = 53, 
                      output_file = "btcusdc_pca_components_12h_53_proper_split_2.csv")

cat("\n\nPCA analysis with proper train/validation/test splits complete.\n")

