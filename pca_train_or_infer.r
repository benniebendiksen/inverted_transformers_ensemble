# PCA Analysis and Export for Cryptocurrency Time Series with Proper Train/Val/Test Split
# Enhanced with support for both percentage-based and fixed-position splitting
library(readr)
library(tidyverse)
library(stats)

# Set the file path
# file_path <- "/Users/bendiksen/Desktop/inverted_transformers_ensemble/binance_futures_historical_data/btcusdt_12h_historical_reduced_python_processed.csv"
# file_path <- "/Users/bendiksen/Desktop/inverted_transformers_ensemble/binance_futures_historical_data/btcusdt_12h_4h_complete_top_100_features_light_gbm.csv"
#file_path <- "/Users/bendiksen/Desktop/inverted_transformers_ensemble/binance_futures_historical_data/btcusdt_12h_4h_complete_top_150_features_light_gbm.csv"


#file_path <- "/Users/bendiksen/Desktop/iTransformer/optimized_features_light/btcusdt_12h_4h_complete_reattempt_top_150_features_light_gbm.csv"
#file_path <- "/Users/bendiksen/Desktop/iTransformer/optimized_features_light/btcusdt_12h_4h_complete_top_150_features_light_gbm_baseline.csv"

#file_path <- "/Users/bendiksen/Desktop/iTransformer/optimized_features_light/btcusdt_12h_4h_complete_reattempt_reordered.csv"

#file_path <- "/Users/bendiksen/Desktop/iTransformer/optimized_features_light/btcusdt_12h_4h_april_15_top_150_features_light_gbm_reduced.csv"
#file_path <- "/Users/bendiksen/Desktop/iTransformer/optimized_features_light/btcusdt_12h_4h_april_15_top_150_features_light_gbm_reduced_extended_14_fixed_sizes.csv"
#file_path <- "/Users/bendiksen/Desktop/iTransformer/optimized_features_light/btcusdt_12h_4h_april_15_top_150_features_light_gbm_reduced_extended_28_fixed_sizes.csv"
#file_path <- "/Users/bendiksen/Desktop/iTransformer/optimized_features_light/btcusdt_12h_4h_april_15_top_150_features_light_gbm_reduced_extended_14_double_fixed_sizes.csv"
#file_path <- "/Users/bendiksen/Desktop/iTransformer/optimized_features_light/btcusdt_12h_historical_reduced_python_processed_1_2_1_old_reattempt_top_80_features_light_gbm.csv"

#file_path <- "/Users/bendiksen/Desktop/iTransformer/dataset/logits/btcusdt_12h_features_april_15_reduced.csv"
#file_path <- "/Users/bendiksen/Desktop/iTransformer/dataset/logits/btcusdt_12h_features_numeriques_april_15_reduced.csv"

#file_path = "/Users/bendiksen/Desktop/iTransformer/dataset/logits/btcusdt_12h_features_numeriques_april_15_reduced_top_80_features_light_gbm.csv"

#file_path <- "/Users/bendiksen/Desktop/iTransformer/dataset/logits/btcusdt_12h_historical_reduced_python_processed_1_2_1_old_reattempt.csv"
#file_path <- "/Users/bendiksen/Desktop/inverted_transformers_ensemble/binance_futures_historical_data/btcusdt_12h_python_processed_reduced.csv"
#file_path <- "/Users/bendiksen/Desktop/inverted_transformers_ensemble/binance_futures_historical_data/btcusdt_12h_historical_reduced_python_processed_1_2_1_old_reattempt_2.csv"
#file_path <- "/Users/bendiksen/Desktop/inverted_transformers_ensemble/binance_futures_historical_data/btcusdt_12h_reduced_python_processed_1_2_1_march_17.csv"
#file_path <- "/Users/bendiksen/Desktop/inverted_transformers_ensemble/binance_futures_historical_data/btcusdt_12h_reduced_python_processed_1_2_1_april_15.csv"
#file_path <- "/Users/bendiksen/Desktop/inverted_transformers_ensemble/binance_futures_historical_data/btcusdt_12h_reduced_python_processed_1_2_1_april_15_baseline_set_sizes.csv"
#file_path <- "/Users/bendiksen/Desktop/inverted_transformers_ensemble/binance_futures_historical_data/btcusdt_12h_reduced_python_processed_1_2_1_april_15.csv"
file_path <- "/Users/bendiksen/Desktop/inverted_transformers_ensemble/binance_futures_historical_data/btcusdt_12h_python_processed_1_2_1_april_15.csv"
#===============================================================================
# Data Loading and Preprocessing Functions
#===============================================================================

# Function to read and prepare the data with explicit timestamp handling
load_data <- function(file_path) {
  # Read the CSV file
  df <- read_csv(file_path)
  
  # Check if timestamp column exists and try to convert it with proper error handling
  if("timestamp" %in% colnames(df)) {
    # First check what type of data we're dealing with
    cat("First few values in timestamp column:\n")
    print(head(df$timestamp))
    
    # Try to handle different timestamp formats
    tryCatch({
      # Check if it's already a datetime object
      if(!inherits(df$timestamp, "POSIXct")) {
        # Try parsing with explicit format if it's a character
        if(is.character(df$timestamp)) {
          # Check format of first timestamp to determine proper parsing
          first_ts <- df$timestamp[1]
          
          if(grepl("^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}$", first_ts)) {
            # Standard format: "YYYY-MM-DD HH:MM:SS"
            df$timestamp <- as.POSIXct(df$timestamp, format="%Y-%m-%d %H:%M:%S")
          } else if(grepl("^\\d{1,2}/\\d{1,2}/\\d{4}\\s+\\d{1,2}:\\d{2}:\\d{2}\\s+[AP]M$", first_ts)) {
            # Format: "M/D/YYYY HH:MM:SS AM/PM"
            cat("Detected format: M/D/YYYY HH:MM:SS AM/PM\n")
            df$timestamp <- as.POSIXct(df$timestamp, format="%m/%d/%Y %I:%M:%S %p")
          } else if(grepl("^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}", first_ts)) {
            # ISO format: "YYYY-MM-DDThh:mm:ss"
            df$timestamp <- as.POSIXct(df$timestamp, format="%Y-%m-%dT%H:%M:%S")
          } else if(grepl("^\\d+$", first_ts)) {
            # Unix timestamp (seconds since epoch)
            df$timestamp <- as.POSIXct(as.numeric(df$timestamp), origin="1970-01-01")
          } else {
            # If format is unknown, try default parsing first
            cat("Attempting to auto-detect timestamp format...\n")
            df$timestamp <- as.POSIXct(df$timestamp)
          }
        } else if(is.numeric(df$timestamp)) {
          # Numeric timestamps are assumed to be Unix timestamps
          df$timestamp <- as.POSIXct(df$timestamp, origin="1970-01-01")
        }
      }
      
      cat("Timestamp conversion successful. Sample values:\n")
      print(head(df$timestamp))
      
    }, error = function(e) {
      cat("Error converting timestamps:", e$message, "\n")
      cat("Will proceed with timestamp column as-is.\n")
    })
  }
  
  # Check for infinite values and replace with NA
  df <- df %>% mutate_all(~ifelse(is.infinite(.), NA, .))
  
  return(df)
}

# ENHANCED: Modified to support both percentage-based and fixed-position splitting approaches
split_data_chronologically <- function(df, 
                                       # Percentage-based parameters  
                                       train_ratio = NULL, valid_ratio = NULL, test_ratio = NULL,
                                       # Fixed position parameters
                                       fixed_val_start = NULL, val_size = NULL) {
  
  # Determine which method to use based on provided parameters
  use_percentage = !is.null(train_ratio) && !is.null(valid_ratio)
  use_fixed_indices = !is.null(fixed_val_start) && !is.null(val_size)
  
  if (!use_percentage && !use_fixed_indices) {
    stop("Must provide either percentage parameters (train_ratio, valid_ratio) or fixed index parameters (fixed_val_start, val_size)")
  }
  
  if (use_percentage && use_fixed_indices) {
    warning("Both percentage and fixed index parameters provided. Using fixed index parameters.")
    use_percentage = FALSE
  }
  
  # Ensure data is ordered chronologically
  if("timestamp" %in% colnames(df)) {
    # Check if timestamp is a valid datetime column
    if(inherits(df$timestamp, "POSIXct")) {
      df <- df %>% arrange(timestamp)
    } else {
      # If timestamp isn't a proper datetime, just use row order
      cat("Warning: timestamp not converted to datetime. Using row order for chronological splitting.\n")
    }
  }
  
  # Calculate total number of rows
  n <- nrow(df)
  
  if (use_percentage) {
    # Percentage-based approach (original method)
    n_train <- floor(n * train_ratio)
    n_valid <- floor(n * valid_ratio)
    n_test <- n - n_train - n_valid  # Remainder goes to test
    
    # Create the splits
    train_idx <- 1:n_train
    valid_idx <- (n_train + 1):(n_train + n_valid)
    test_idx <- (n_train + n_valid + 1):n
    
    cat("Data split chronologically using percentage-based method:\n")
    cat("  Training set:   ", length(train_idx), "rows (", train_ratio*100, "%)\n")
    cat("  Validation set: ", length(valid_idx), "rows (", valid_ratio*100, "%)\n")
    cat("  Test set:       ", length(test_idx), "rows (", test_ratio*100, "%)\n")
    
  } else {
    # Fixed position approach (new method, similar to first script)
    # Calculate training size and test start
    train_size <- fixed_val_start - 1
    test_start <- fixed_val_start + val_size
    
    # Create the splits
    train_idx <- 1:train_size
    valid_idx <- fixed_val_start:(fixed_val_start + val_size - 1)
    test_idx <- test_start:n
    
    # Calculate percentages for informational purposes
    train_pct <- length(train_idx) / n * 100
    valid_pct <- length(valid_idx) / n * 100
    test_pct <- length(test_idx) / n * 100
    
    cat("Data split chronologically using fixed position method:\n")
    cat("  Training set:   ", length(train_idx), "rows (", round(train_pct, 2), "%) - rows 1 to", max(train_idx), "\n")
    cat("  Validation set: ", length(valid_idx), "rows (", round(valid_pct, 2), "%) - rows", min(valid_idx), "to", max(valid_idx), "\n")
    cat("  Test set:       ", length(test_idx), "rows (", round(test_pct, 2), "%) - rows", min(test_idx), "to", n, "\n")
  }
  
  # Create split indicators
  split_indicators <- rep(NA, n)
  split_indicators[train_idx] <- "train"
  split_indicators[valid_idx] <- "val"
  split_indicators[test_idx] <- "test"
  
  # Add split column to the dataframe
  df$split <- split_indicators
  
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

# Function to identify and remove highly correlated features - using only training data, makes call to save resulting dataset
remove_highly_correlated <- function(df, train_idx, file_path, preserved_cols = NULL, threshold = 0.95, write_csv = TRUE) {
  # Calculate the correlation matrix using only training data
  train_data <- df[train_idx, ] %>% select_if(is.numeric)
  
  # Use pairwise.complete.obs to handle NA values
  cor_matrix <- cor(train_data, use = "pairwise.complete.obs")
  
  # Check for NA values in correlation matrix
  if(any(is.na(cor_matrix))) {
    cat("Warning: Some correlations could not be computed due to missing values.\n")
    cat("Setting NA correlations to 0 to proceed with analysis.\n")
    cor_matrix[is.na(cor_matrix)] <- 0
  }
  
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
  
  # Create reduced dataframe
  df_reduced <- df %>% select(-one_of(cols_to_remove))
  cat("Dataset reduced to", ncol(df_reduced), "columns\n")
  
  # Write reduced CSV if requested
#  if (write_csv && !is.null(file_path)) {
#    new_file_path <- write_reduced_csv(df_reduced, file_path, cols_to_remove, preserved_cols)
#    cat("Reduced dataset written to:", new_file_path, "\n")
 # }
  
  # Return data without highly correlated columns (applied to all data)
  return(list(
    data = df_reduced,
    removed = cols_to_remove
  ))
}

# Function to identify and remove zero variance columns - using only training data
remove_zero_variance <- function(df, train_idx) {
  # Calculate variance for all numeric columns in training data only
  train_data <- df[train_idx, ] %>% select_if(is.numeric)
  variances <- apply(train_data, 2, var, na.rm = TRUE)
  
  # Identify columns with near-zero variance
  zero_var_cols <- names(which(variances < 1e-10 | is.na(variances)))
  
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

# Function to save preprocessed data with preserved columns before PCA
save_preprocessed_data <- function(processed_data, preserved_cols, file_path, suffix = "_corr_removed") {
  # Extract file path components
  dir_path <- dirname(file_path)
  file_name <- basename(file_path)
  
  # Split filename into name and extension
  file_parts <- strsplit(file_name, "\\.")[[1]]
  base_name <- file_parts[1]
  extension <- file_parts[length(file_parts)]
  
  # Create the new filename with suffix
  new_file_name <- paste0(base_name, suffix, ".", extension)
  new_file_path <- file.path(dir_path, new_file_name)
  
  # Create a copy of the processed dataframe
  output_df <- processed_data
  
  # Add timestamp as first column if it exists and rename to "date" for consistency
  if (!is.null(preserved_cols$timestamp)) {
    output_df <- cbind(date = preserved_cols$timestamp, output_df)
    # Note: using "date" as column name to match the PCA output format
  }
  
  # Add close price if it exists
  if (!is.null(preserved_cols$close)) {
    output_df$close <- preserved_cols$close
  }
  
  # Add direction column if it exists
  if (!is.null(preserved_cols$direction)) {
    output_df$direction <- preserved_cols$direction
  }
  
  # Add split column if it exists
  if (!is.null(preserved_cols$split)) {
    output_df$split <- preserved_cols$split
  }
  
  # Write the dataframe to CSV
  write_csv(output_df, new_file_path)
  
  # Calculate file stats for reporting
  if (file.exists(file_path) && file.exists(new_file_path)) {
    original_size <- file.size(file_path) / (1024 * 1024)  # Size in MB
    new_size <- file.size(new_file_path) / (1024 * 1024)  # Size in MB
    
    cat("\n=== WRITING PREPROCESSED CSV FILE ===\n")
    cat("Original file:", file_path, "\n")
    cat("Preprocessed file:", new_file_path, "\n")
    cat("Data dimensions:", nrow(output_df), "rows ×", ncol(output_df), "columns\n")
    
    if (!is.null(preserved_cols)) {
      preserved_count <- sum(!sapply(preserved_cols, is.null))
      
      # Report on specific columns
      if (!is.null(preserved_cols$timestamp)) {
        cat("Added timestamp column (renamed to 'date')\n")
      }
      if (!is.null(preserved_cols$close)) {
        cat("Added close price column\n")
      }
      if (!is.null(preserved_cols$direction)) {
        cat("Added price direction column\n")
      }
      if (!is.null(preserved_cols$split)) {
        cat("Added train/validation/test split information\n")
      }
      
      # Print split information if available
      if (!is.null(preserved_cols$split)) {
        split_counts <- table(output_df$split)
        cat("\nData split summary:\n")
        print(split_counts)
      }
      
      # Print direction distribution if available
      if (!is.null(preserved_cols$direction)) {
        direction_counts <- table(output_df$direction)
        cat("\nDirection distribution:\n")
        print(direction_counts)
        cat("Percentage up:", round(direction_counts["1"]/sum(direction_counts)*100, 2), "%\n")
      }
    }
    
    cat("Original file size:", round(original_size, 2), "MB\n")
    cat("Preprocessed file size:", round(new_size, 2), "MB\n")
  }
  
  return(new_file_path)
}

#===============================================================================
# PCA Analysis Function - Modified to Respect Train/Val/Test Split
#===============================================================================

# Preprocess and prepare data for PCA - now respecting data splits
preprocess_for_pca <- function(df, train_idx, file_path = NULL) {
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
  reduced <- remove_highly_correlated(df, train_idx, file_path, preserved_cols)
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

# Function to write a reduced CSV after correlation analysis
write_reduced_csv <- function(df, original_file_path, removed_columns, preserved_cols = NULL, suffix = "_corr_reduced") {
  # Extract file path components
  dir_path <- dirname(original_file_path)
  file_name <- basename(original_file_path)
  
  # Split filename into name and extension
  file_parts <- strsplit(file_name, "\\.")[[1]]
  base_name <- file_parts[1]
  extension <- file_parts[length(file_parts)]
  
  # Create the new filename with suffix
  new_file_name <- paste0(base_name, suffix, ".", extension)
  new_file_path <- file.path(dir_path, new_file_name)
  
  # Create a copy of the dataframe for output
  output_df <- df
  
  # Add preserved columns back to the dataframe if provided
  if (!is.null(preserved_cols)) {
    # Add timestamp as first column if it exists
    if (!is.null(preserved_cols$timestamp)) {
      output_df <- cbind(timestamp = preserved_cols$timestamp, output_df)
    }
    
    # Add close price if it exists
    if (!is.null(preserved_cols$close)) {
      output_df$close <- preserved_cols$close
    }
    
    # Add direction column if it exists
    if (!is.null(preserved_cols$direction)) {
      output_df$direction <- preserved_cols$direction
    }
    
    # Add split column if it exists
    if (!is.null(preserved_cols$split)) {
      output_df$split <- preserved_cols$split
    }
  }
  
  # Write the dataframe to CSV
  write_csv(output_df, new_file_path)
  
  # Calculate file stats for reporting
  if (file.exists(original_file_path) && file.exists(new_file_path)) {
    original_size <- file.size(original_file_path) / (1024 * 1024)  # Size in MB
    new_size <- file.size(new_file_path) / (1024 * 1024)  # Size in MB
    size_reduction <- (1 - new_size / original_size) * 100
    
    cat("\n=== WRITING REDUCED CSV FILE ===\n")
    cat("Original file:", original_file_path, "\n")
    cat("Reduced file:", new_file_path, "\n")
    cat("Original dimensions:", nrow(df), "rows ×", ncol(df) + length(removed_columns), "columns (before adding preserved columns)\n")
    cat("Reduced dimensions:", nrow(output_df), "rows ×", ncol(output_df), "columns (with preserved columns)\n")
    cat("Removed", length(removed_columns), "highly correlated columns\n")
    
    if (!is.null(preserved_cols)) {
      preserved_count <- sum(!sapply(preserved_cols, is.null))
      cat("Added back", preserved_count, "preserved columns (timestamp, close, direction, split)\n")
    }
    
    cat("Original file size:", round(original_size, 2), "MB\n")
    cat("Reduced file size:", round(new_size, 2), "MB\n")
    cat("Size reduction:", round(size_reduction, 2), "%\n")
  }
  
  return(new_file_path)
}

# Function to perform PCA - fit on training data only, then transform all data
analyze_pca_components <- function(processed_data, train_idx) {
  # Select only numeric columns
  numeric_df <- processed_data %>% select_if(is.numeric)
  
  # Compute scaling parameters from training data only
  train_center <- colMeans(numeric_df[train_idx, ], na.rm = TRUE)
  train_scale <- apply(numeric_df[train_idx, ], 2, sd, na.rm = TRUE)
  
  # Handle any zero scale values
  zero_scale_cols <- names(which(train_scale == 0 | is.na(train_scale)))
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
  
  if(any(cum_variance >= 0.98)) {
    n_95 <- which(cum_variance >= 0.98)[1]
    cat("\n", n_95, "components needed to explain 98% of variance\n")
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
# Main Analysis Execution Function - Enhanced With Flexible Split Options
#===============================================================================

# ENHANCED: Modified to support both percentage-based and fixed-position splitting approaches
# ENHANCED: Modified to support both percentage-based and fixed-position splitting approaches
# Also saves preprocessed data before PCA analysis
run_pca_analysis <- function(file_path, 
                             # Percentage-based parameters 
                             train_ratio = NULL, valid_ratio = NULL, test_ratio = NULL,
                             # Fixed position parameters
                             fixed_val_start = NULL, val_size = NULL,
                             # Optional parameters
                             save_preprocessed = TRUE) {
  
  # Validate parameters - must have either percentage or fixed indices
  use_percentage = !is.null(train_ratio) && !is.null(valid_ratio)
  use_fixed_indices = !is.null(fixed_val_start) && !is.null(val_size)
  
  if (!use_percentage && !use_fixed_indices) {
    stop("Must provide either percentage parameters (train_ratio, valid_ratio, test_ratio) or fixed index parameters (fixed_val_start, val_size)")
  }
  
  if (use_percentage && use_fixed_indices) {
    warning("Both percentage and fixed index parameters provided. Using fixed index parameters.")
    use_percentage = FALSE
  }
  
  # Load data
  cat("Loading data...\n")
  df <- load_data(file_path)
  cat("Dataset loaded:", nrow(df), "rows,", ncol(df), "columns\n")
  
  # Split data chronologically with flexible approach
  cat("\n=== SPLITTING DATA CHRONOLOGICALLY ===\n")
  if (use_percentage) {
    split_data <- split_data_chronologically(
      df = df, 
      train_ratio = train_ratio, 
      valid_ratio = valid_ratio, 
      test_ratio = test_ratio,
      fixed_val_start = NULL, 
      val_size = NULL
    )
  } else {
    split_data <- split_data_chronologically(
      df = df, 
      train_ratio = NULL, 
      valid_ratio = NULL, 
      test_ratio = NULL,
      fixed_val_start = fixed_val_start, 
      val_size = val_size
    )
  }
  
  df_split <- split_data$data
  train_idx <- split_data$train_idx
  valid_idx <- split_data$valid_idx
  test_idx <- split_data$test_idx
  
  # Preprocess data for PCA - using only training data for fitting
  cat("\n=== PREPROCESSING DATA (USING TRAIN DATA FOR PARAMETERS) ===\n")
  preprocessed <- preprocess_for_pca(df_split, train_idx, file_path)
  
  # Save preprocessed data before PCA if requested
  if (save_preprocessed) {
    cat("\n=== SAVING PREPROCESSED DATA BEFORE PCA ===\n")
    preprocessed_file <- save_preprocessed_data(
      processed_data = preprocessed$processed_data,
      preserved_cols = preprocessed$preserved_cols,
      file_path = file_path
    )
    cat("Preprocessed data saved to:", preprocessed_file, "\n")
  }
  
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
  
  
  # Add timestamp as first column if it exists
  else if (!is.null(preserved_cols$time)) {
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
  
  # Print direction distribution if available
  if (!is.null(preserved_cols$direction)) {
    direction_counts <- table(final_data$direction)
    cat("\nDirection distribution in exported data:\n")
    print(direction_counts)
    cat("Percentage up:", round(direction_counts["1"]/sum(direction_counts)*100, 2), "%\n")
  }
  
  return(final_data)
}

# Export multiple versions with different component counts
export_multiple_versions <- function(results, component_counts, base_filename = "btcusdt_pca_components") {
  exported_datasets <- list()
  
  for (n in component_counts) {
    output_file <- paste0(base_filename, "_", n, ".csv")
    exported_datasets[[as.character(n)]] <- export_pca_components(results, n, output_file)
  }
  
  return(exported_datasets)
}

#===============================================================================
# Script Execution with Flexible Split Options
#===============================================================================

# Example 1: Run the analysis with percentage-based split

results <- run_pca_analysis(
   file_path = file_path, 
   train_ratio = 0.88, 
   valid_ratio = 0.07, 
   test_ratio = 0.05
 )

# Example 2: Run the analysis with fixed position split
#results <- run_pca_analysis(
#  file_path = file_path,
#  fixed_val_start = 3551,    # Validation starts at row 3552,
#  val_size = 282             # Validation set is 282 rows
#)


# Export dataset with specified number of components
export_pca_components(results, n_components = 46, 
                      output_file = "pca_components_btcusdt_12h_46_07_05_lance_seed_april_15.csv") 

cat("\n\nPCA analysis with flexible splitting options complete.\n")


