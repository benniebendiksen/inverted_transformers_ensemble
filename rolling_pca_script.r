# Rolling Window PCA Analysis for Cryptocurrency Time Series
# This script implements a sliding window approach for time series forecasting
# with proper sequence handling for test samples and flexible split options

library(readr)
library(tidyverse)
library(stats)

# Set the file path - use the one from the original script
file_path <- "/Users/bendiksen/Desktop/inverted_transformers_ensemble/binance_futures_historical_data/btcusdt_12h_historical.csv"

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
# PCA Analysis Functions
#===============================================================================

# Preprocess and prepare data for PCA - now respecting data splits
preprocess_for_pca <- function(df, train_idx) {
  # Extract and save non-numeric columns we want to keep
  preserved_cols <- list()
  
  # Updated list of variables to preserve - now includes 'direction'
  preserved_vars <- c("timestamp", "close", "direction", "split")
  
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
  
  # Now transform all data using the PCA fitted on training data
  all_transformed <- predict(pca_result, scaled_data)
  
  # Return variance data and PCA results for further analysis
  return(list(
    pca_result = pca_result,
    all_transformed = all_transformed,
    scaling_params = scaling_params,
    variance_df = variance_df,
    selected_components = list(
      kaiser = sum(variance > 1),
      var_70 = if(exists("n_70")) n_70 else NA,
      var_80 = if(exists("n_80")) n_80 else NA,
      var_90 = if(exists("n_90")) n_90 else NA,
      var_95 = if(exists("n_95")) n_95 else NA
    ),
    removed_cols = if(exists("zero_scale_cols")) zero_scale_cols else character(0)
  ))
}

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
  
  # Add direction column if it exists
  if (!is.null(preserved_cols$direction)) {
    cat("Adding price direction column\n")
    final_data$direction <- preserved_cols$direction
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
    output_file <- paste0("btcusdt_pca_components_", n_components, ".csv")
  }
  
  # Check if the output_file already contains a path
  if (dirname(output_file) != ".") {
    # If output_file already has a path, use it directly
    export_path <- output_file
  } else {
    # Otherwise, prepend the directory of file_path
    export_path <- file.path(dirname(file_path), output_file)
  }
  
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

#===============================================================================
# New Rolling Window Functions for Generating Multiple PCA Datasets
#===============================================================================

# Function to create custom train/val/test split with support for both methods
create_rolling_split <- function(df, 
                                 # Percentage-based parameters
                                 base_train_pct = NULL, base_val_pct = NULL,
                                 # Fixed position parameters
                                 fixed_val_start = NULL, val_size = NULL,
                                 # Common parameters
                                 test_samples = 10, sequence_length = 98, sliding_offset = 1, window_index = 0) {
  
  # Ensure data is ordered chronologically (if timestamp is available)
  if("timestamp" %in% colnames(df)) {
    df <- df %>% arrange(timestamp)
  }
  
  # Calculate total number of rows
  n <- nrow(df)
  
  # Determine which method to use based on provided parameters
  use_percentage = !is.null(base_train_pct) && !is.null(base_val_pct)
  use_fixed_indices = !is.null(fixed_val_start) && !is.null(val_size)
  
  if (!use_percentage && !use_fixed_indices) {
    stop("Must provide either percentage parameters (base_train_pct, base_val_pct) or fixed index parameters (fixed_val_start, val_size)")
  }
  
  if (use_percentage && use_fixed_indices) {
    warning("Both percentage and fixed index parameters provided. Using fixed index parameters.")
    use_percentage = FALSE
  }
  
  # Calculate split based on selected method
  if (use_percentage) {
    # Percentage-based approach
    base_train_size <- floor(n * base_train_pct)
    base_val_size <- floor(n * base_val_pct)
    
    # For each window, increment train and val sets by exactly test_samples rows
    current_train_size <- base_train_size + (window_index * test_samples)
    current_val_size <- base_val_size + (window_index * test_samples)
    
    # Calculate validation start and test start
    val_start <- current_train_size + 1
    test_start <- current_train_size + current_val_size
  } else {
    # Fixed index approach
    # For each window, shift val and test by test_samples rows
    current_val_start <- fixed_val_start + (window_index * test_samples)
    current_val_end <- current_val_start + val_size - 1
    
    # Calculate training size and test start
    current_train_size <- current_val_start - 1
    test_start <- current_val_end + 1
    current_val_size <- val_size
  }
  
  # Check if we have enough data left for at least one full sequence
  if (test_start + sequence_length > n) {
    cat("Cannot create window", window_index + 1, "- insufficient data for even one sequence\n")
    cat("  Test would start at row:", test_start + 1, "\n")
    cat("  Minimum needed for one sequence:", test_start + sequence_length, "\n")
    cat("  Dataset has", n, "rows\n")
    return(NULL)  # No more windows available
  }
  
  # Calculate how many complete samples we can fit
  remaining_rows <- n - test_start
  max_samples <- floor((remaining_rows - sequence_length) / sliding_offset) + 1
  max_samples <- max(1, max_samples)  # Ensure at least 1 sample
  
  # If we have fewer rows than needed for requested samples, adjust
  actual_samples <- min(test_samples, max_samples)
  
  # Calculate actual test size for the available samples
  actual_test_size <- sequence_length + (sliding_offset * (actual_samples - 1))
  test_end <- test_start + actual_test_size - 1
  
  # Print diagnostic information
  if (actual_samples < test_samples) {
    cat("Notice: Final window will use", actual_samples, "test samples instead of", test_samples, "\n")
    cat("  - Test starts at row:", test_start + 1, "\n")
    cat("  - Test ends at row:", test_end + 1, "\n")
    cat("  - Dataset has", n, "rows\n")
  }
  
  # Define indices for each split - ensure we're using 1-based indexing
  if (use_percentage) {
    train_idx <- 1:current_train_size
    val_idx <- (current_train_size + 1):(current_train_size + current_val_size)
  } else {
    train_idx <- 1:current_train_size
    val_idx <- current_val_start:current_val_end
  }
  test_idx <- (test_start + 1):(test_end + 1)
  
  # Verify indices are within bounds
  if (max(test_idx) > n) {
    cat("Warning: Adjusting test indices to fit within dataset bounds\n")
    test_idx <- test_idx[test_idx <= n]
    actual_samples <- ceiling(length(test_idx) / sliding_offset)
  }
  
  # Create split indicators - initialize with "unknown" to avoid NAs
  split_indicators <- rep("unknown", nrow(df))
  
  # Assign split indicators
  split_indicators[train_idx] <- "train"
  split_indicators[val_idx] <- "val"
  split_indicators[test_idx] <- "test"
  
  # Check for any rows that might be between splits (in case of gaps)
  if (use_percentage) {
    # Check for gap between validation and test
    gap_start <- current_train_size + current_val_size + 1
    gap_end <- test_start
    if (gap_start <= gap_end) {
      # Assign gap rows to validation
      split_indicators[gap_start:gap_end] <- "val"
      # Update val_idx to include gap
      val_idx <- c(val_idx, gap_start:gap_end)
      current_val_size <- length(val_idx)
    }
  } else {
    # Check for gap between validation and test
    gap_start <- current_val_end + 1
    gap_end <- test_start
    if (gap_start <= gap_end) {
      # Assign gap rows to validation
      split_indicators[gap_start:gap_end] <- "val"
      # Update val_idx to include gap
      val_idx <- c(val_idx, gap_start:gap_end)
    }
  }
  
  # Add split column to dataframe
  df$split <- split_indicators
  
  cat("Rolling Window #", window_index + 1, ":\n")
  cat("  Training set:   ", length(train_idx), "rows (1 to", max(train_idx), ")\n")
  cat("  Validation set: ", length(val_idx), "rows (", min(val_idx), "to", max(val_idx), ")\n")
  cat("  Test set:       ", length(test_idx), "rows (", min(test_idx), "to", max(test_idx), ")\n")
  cat("    - Sequence length:  ", sequence_length, "rows\n")
  cat("    - Sliding offset:   ", sliding_offset, "row(s) per additional sample\n")
  cat("    - Test samples:     ", actual_samples, "samples\n")
  cat("  Total data used:", length(train_idx) + length(val_idx) + length(test_idx), "rows\n")
  
  return(list(
    data = df,
    train_idx = train_idx,
    valid_idx = val_idx,
    test_idx = test_idx,
    window_index = window_index,
    sequence_length = sequence_length,
    sliding_offset = sliding_offset,
    test_samples = actual_samples  # Return the actual number of samples
  ))
}

# Function to run PCA analysis with custom split
run_rolling_pca_analysis <- function(df, split_data) {
  # Extract indices
  train_idx <- split_data$train_idx
  
  # Preprocess data for PCA
  cat("\n=== PREPROCESSING DATA (USING TRAIN DATA FOR PARAMETERS) ===\n")
  preprocessed <- preprocess_for_pca(split_data$data, train_idx)
  
  # Run PCA analysis
  cat("\n=== PCA COMPONENT ANALYSIS (FIT ON TRAIN DATA ONLY) ===\n")
  pca_results <- analyze_pca_components(preprocessed$processed_data, train_idx)
  
  # Return results
  return(list(
    split_data = split_data$data,
    train_idx = train_idx,
    valid_idx = split_data$valid_idx,
    test_idx = split_data$test_idx,
    processed_data = preprocessed$processed_data,
    preserved_cols = preprocessed$preserved_cols,
    removed_features = preprocessed$removed,
    pca_results = pca_results,
    window_index = split_data$window_index,
    sequence_length = split_data$sequence_length,
    sliding_offset = split_data$sliding_offset,
    test_samples = split_data$test_samples
  ))
}

# Main function to run rolling window PCA analysis with flexible split options
run_rolling_window_pca <- function(file_path, 
                                   # Percentage-based parameters 
                                   base_train_pct = NULL, base_val_pct = NULL,
                                   # Fixed position parameters
                                   fixed_val_start = NULL, val_size = NULL,
                                   # Common parameters
                                   test_samples = 10, sequence_length = 98, sliding_offset = 1,
                                   n_components = 55, max_windows = 100) {
  
  # Validate parameters - must have either percentage or fixed indices
  use_percentage = !is.null(base_train_pct) && !is.null(base_val_pct)
  use_fixed_indices = !is.null(fixed_val_start) && !is.null(val_size)
  
  if (!use_percentage && !use_fixed_indices) {
    stop("Must provide either percentage parameters (base_train_pct, base_val_pct) or fixed index parameters (fixed_val_start, val_size)")
  }
  
  if (use_percentage && use_fixed_indices) {
    warning("Both percentage and fixed index parameters provided. Using fixed index parameters.")
    use_percentage = FALSE
  }
  
  # Load data
  cat("Loading data...\n")
  df <- load_data(file_path)
  cat("Dataset loaded:", nrow(df), "rows,", ncol(df), "columns\n")
  
  # Parse the file path to get the base directory
  base_dir <- dirname(file_path)
  
  # Create output directory for PCA results
  output_dir <- file.path(base_dir, "pca_rolling_windows")
  
  # Create the directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    cat("Creating output directory:", output_dir, "\n")
    dir.create(output_dir, showWarnings = TRUE, recursive = TRUE)
  } else {
    cat("Using existing output directory:", output_dir, "\n")
  }
  
  # Process windows
  window_index <- 0
  total_rows <- nrow(df)
  
  # Calculate actual test size using sliding window approach
  actual_test_size <- sequence_length + (sliding_offset * (test_samples - 1))
  
  # Calculate and display initial split info based on selected method
  if (use_percentage) {
    base_train_size <- floor(total_rows * base_train_pct)
    base_val_size <- floor(total_rows * base_val_pct)
    
    # Calculate maximum possible windows
    available_rows <- total_rows - (base_train_size + base_val_size + actual_test_size)
    max_possible_windows <- floor(available_rows / test_samples) + 1
    
    cat("Dataset has", total_rows, "rows\n")
    cat("Using percentage-based split:\n")
    cat("  - Training set:    ", base_train_pct * 100, "% (", base_train_size, "rows)\n")
    cat("  - Validation set:  ", base_val_pct * 100, "% (", base_val_size, "rows)\n")
    cat("  - First test starts at row:", base_train_size + base_val_size + 1, "\n")
  } else {
    # Calculate maximum possible windows for fixed approach
    first_test_start <- fixed_val_start + val_size
    available_rows <- total_rows - first_test_start - actual_test_size + 1
    max_possible_windows <- floor(available_rows / test_samples) + 1
    
    cat("Dataset has", total_rows, "rows\n")
    cat("Using fixed index split:\n")
    cat("  - Training set ends at:  ", fixed_val_start - 1, "\n")
    cat("  - Validation starts at:  ", fixed_val_start, "\n")
    cat("  - Validation size:       ", val_size, "rows\n")
    cat("  - First test starts at:  ", first_test_start, "\n")
  }
  
  cat("Test set configuration:\n")
  cat("  - Sequence length:  ", sequence_length, "rows\n")
  cat("  - Sliding offset:   ", sliding_offset, "row(s) per additional sample\n")
  cat("  - Test samples:     ", test_samples, "samples\n")
  cat("  - Total test size:  ", actual_test_size, "rows\n")
  cat("Window shifting: Each window shifts by", test_samples, "rows\n")
  cat("Maximum possible windows:", max_possible_windows, "\n")
  
  while (TRUE) {
    # Check if we've reached max windows
    if (window_index >= max_windows) {
      cat("Reached maximum number of windows (", max_windows, ").\n")
      break
    }
    
    # Create split for this window
    cat("\n\n=========================================================\n")
    cat("PROCESSING WINDOW", window_index + 1, "\n")
    cat("=========================================================\n\n")
    
    # Call the appropriate create_rolling_split with the right parameters
    if (use_percentage) {
      split_data <- create_rolling_split(
        df = df, 
        base_train_pct = base_train_pct, 
        base_val_pct = base_val_pct,
        fixed_val_start = NULL, 
        val_size = NULL,
        test_samples = test_samples, 
        sequence_length = sequence_length, 
        sliding_offset = sliding_offset, 
        window_index = window_index
      )
    } else {
      split_data <- create_rolling_split(
        df = df, 
        base_train_pct = NULL, 
        base_val_pct = NULL,
        fixed_val_start = fixed_val_start, 
        val_size = val_size,
        test_samples = test_samples, 
        sequence_length = sequence_length, 
        sliding_offset = sliding_offset, 
        window_index = window_index
      )
    }
    
    # Check if we've run out of data
    if (is.null(split_data)) {
      cat("No more data available for additional windows.\n")
      break
    }
    
    # Run PCA analysis for this window
    results <- run_rolling_pca_analysis(df, split_data)
    
    # Export results
    output_file <- paste0("btcusdt_pca_components_12h_60_07_05_", n_components, "_window_", window_index + 1, ".csv")
    output_path <- file.path(output_dir, output_file)
    
    cat("\n=== EXPORTING PCA RESULTS FOR WINDOW", window_index + 1, "===\n")
    export_pca_components(results, n_components, output_path)
    
    # Track PCA parameters for this window (for future reference)
    if (window_index == 0) {
      pca_params_df <- data.frame(
        window = window_index + 1,
        components_70pct = results$pca_results$selected_components$var_70,
        components_80pct = results$pca_results$selected_components$var_80,
        components_90pct = results$pca_results$selected_components$var_90,
        components_95pct = results$pca_results$selected_components$var_95,
        train_size = length(results$train_idx),
        val_size = length(results$valid_idx),
        test_size = length(results$test_idx),
        sequence_length = results$sequence_length,
        sliding_offset = results$sliding_offset,
        test_samples = results$test_samples
      )
    } else {
      pca_params_df <- rbind(pca_params_df, data.frame(
        window = window_index + 1,
        components_70pct = results$pca_results$selected_components$var_70,
        components_80pct = results$pca_results$selected_components$var_80,
        components_90pct = results$pca_results$selected_components$var_90,
        components_95pct = results$pca_results$selected_components$var_95,
        train_size = length(results$train_idx),
        val_size = length(results$valid_idx),
        test_size = length(results$test_idx),
        sequence_length = results$sequence_length,
        sliding_offset = results$sliding_offset,
        test_samples = results$test_samples
      ))
    }
    
    # Move to next window
    window_index <- window_index + 1
  }
  
  # Export PCA parameters summary to the output directory
  summary_path <- file.path(output_dir, "pca_window_parameters_summary.csv")
  write_csv(pca_params_df, summary_path)
  cat("\nSaved PCA parameters summary to:", summary_path, "\n")
  
  cat("\nRolling window PCA analysis complete. Processed", window_index, "windows.\n")
  return(pca_params_df)
}

#===============================================================================
# Script Execution
#===============================================================================

# Example 1: Using percentage-based split
#results_summary <- run_rolling_window_pca(
#  file_path = file_path,
#  base_train_pct = 0.88,     # Using 88% for training
#  base_val_pct = 0.07,       # Using 7% for validation
#  test_samples = 10,
#  sequence_length = 98,
#  sliding_offset = 1,
# n_components = 55,
#  max_windows = 100
#)

# Alternatively, you can use fixed index split:
 results_summary <- run_rolling_window_pca(
   file_path = file_path,
   fixed_val_start = 3558,    # Validation starts at row 3552
   val_size = 282,            # Validation set is 282 rows
   test_samples = 10,
   sequence_length = 97,
   sliding_offset = 1,
   n_components = 60,
   max_windows = 100
 )

# Print summary of windows processed
cat("\n\nRolling Window PCA Analysis Summary:\n")
cat("Total windows processed:", nrow(results_summary), "\n")
cat("Components needed for 90% variance (first window):", results_summary$components_90pct[1], "\n")
cat("Final train size (last window):", results_summary$train_size[nrow(results_summary)], "\n")
cat("Final validation size (last window):", results_summary$val_size[nrow(results_summary)], "\n")
cat("Test set composition:", results_summary$sequence_length[1], "sequence rows +", 
    results_summary$test_samples[1] - 1, "additional rows with offset", results_summary$sliding_offset[1], "\n")

