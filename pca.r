# Optimized PCA Analysis and Export for Cryptocurrency Time Series
library(readr)
library(tidyverse)
library(stats)

# Set the file path
file_path <- "/Users/bendiksen/Desktop/inverted_transformers_ensemble/binance_us_historical_data/btcusdc_15m_historical.csv"

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
  
  # Handle NA values - replace with column medians
  df <- df %>% mutate_all(~ifelse(is.na(.), median(., na.rm = TRUE), .))
  
  return(df)
}

# Function to identify and remove highly correlated features
remove_highly_correlated <- function(df, threshold = 0.95) {
  # Calculate the correlation matrix
  cor_matrix <- cor(df %>% select_if(is.numeric), use = "complete.obs")
  
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
  cat("Top highly correlated pairs:\n")
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
  
  cat("\nRemoving", length(cols_to_remove), "highly correlated features\n")
  
  # Return data without highly correlated columns
  return(list(
    data = df %>% select(-one_of(cols_to_remove)),
    removed = cols_to_remove
  ))
}

# Function to identify and remove zero variance columns
remove_zero_variance <- function(df) {
  # Calculate variance for all numeric columns
  numeric_df <- df %>% select_if(is.numeric)
  variances <- apply(numeric_df, 2, var)
  
  # Identify columns with near-zero variance
  zero_var_cols <- names(which(variances < 1e-10))
  
  if(length(zero_var_cols) > 0) {
    cat("\nRemoving", length(zero_var_cols), "columns with zero or near-zero variance:\n")
    print(zero_var_cols)
    
    # Return data without zero variance columns
    return(list(
      data = df %>% select(-one_of(zero_var_cols)),
      removed = zero_var_cols
    ))
  } else {
    cat("\nNo zero variance columns found.\n")
    return(list(
      data = df,
      removed = character(0)
    ))
  }
}

#===============================================================================
# PCA Analysis Function
#===============================================================================

# Preprocess and prepare data for PCA
preprocess_for_pca <- function(df) {
  # Extract and save non-numeric columns we want to keep
  preserved_cols <- NULL
  if ("timestamp" %in% colnames(df)) {
    preserved_cols$timestamp <- df$timestamp
    df <- df %>% select(-timestamp)
  }
  
  if ("close" %in% colnames(df)) {
    preserved_cols$close <- df$close
    df <- df %>% select(-close)
  }
  
  # Remove highly correlated features
  cat("\n=== REMOVING HIGHLY CORRELATED FEATURES ===\n")
  reduced <- remove_highly_correlated(df)
  df_reduced <- reduced$data
  removed_correlated <- reduced$removed
  cat("Dataset reduced to", ncol(df_reduced), "columns\n")
  
  # Remove zero variance features
  cat("\n=== REMOVING ZERO VARIANCE FEATURES ===\n")
  reduced_var <- remove_zero_variance(df_reduced)
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

# Function to perform PCA and analyze variance explained
analyze_pca_components <- function(processed_data) {
  # Select only numeric columns
  numeric_df <- processed_data %>% select_if(is.numeric)
  
  # Standardize the data (center and scale)
  scaled_data <- scale(numeric_df)
  
  # Store scaling parameters for later use
  scaling_params <- list(
    center = attr(scaled_data, "scaled:center"),
    scale = attr(scaled_data, "scaled:scale")
  )
  
  # Check for any NaN values after scaling
  if(any(is.na(scaled_data))) {
    cat("\nWarning: NaN values found after scaling. This indicates columns with zero variance.\n")
    # Identify and print problematic columns
    col_sd <- apply(numeric_df, 2, sd)
    zero_sd_cols <- names(which(col_sd == 0))
    cat("Columns with zero standard deviation:", paste(zero_sd_cols, collapse=", "), "\n")
    
    # Remove problematic columns from scaled data
    cols_to_keep <- which(!colnames(numeric_df) %in% zero_sd_cols)
    scaled_data <- scaled_data[, cols_to_keep]
    cat("Proceeding with PCA after removing", length(zero_sd_cols), "problematic columns\n")
  }
  
  # Perform PCA
  pca_result <- prcomp(scaled_data, center = TRUE, scale. = TRUE)
  
  # Calculate variance explained
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
  cat("\nExplained variance by principal components:\n")
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
  
  # Return variance data and PCA results for further analysis
  return(list(
    pca_result = pca_result,
    scaling_params = scaling_params,
    variance_df = variance_df,
    selected_components = list(
      kaiser = kaiser_n,
      var_70 = if(exists("n_70")) n_70 else NA,
      var_80 = if(exists("n_80")) n_80 else NA,
      var_90 = if(exists("n_90")) n_90 else NA,
      prop_max = prop_max_n,
      avg_var = avg_var_n
    )
  ))
}

#===============================================================================
# Main Analysis Execution Function
#===============================================================================

# Main execution - perform full PCA analysis
run_pca_analysis <- function(file_path) {
  # Load data
  cat("Loading data...\n")
  df <- load_data(file_path)
  cat("Dataset loaded:", nrow(df), "rows,", ncol(df), "columns\n")
  
  # Preprocess data for PCA
  preprocessed <- preprocess_for_pca(df)
  
  # Run PCA analysis
  cat("\n=== PCA COMPONENT ANALYSIS ===\n")
  pca_results <- analyze_pca_components(preprocessed$processed_data)
  
  # Store all necessary information for later export
  full_results <- list(
    raw_data = df,
    processed_data = preprocessed$processed_data,
    preserved_cols = preprocessed$preserved_cols,
    removed_features = preprocessed$removed,
    pca_results = pca_results
  )
  
  # Save the full results for later use (optional)
  saveRDS(full_results, "pca_full_results.rds")
  
  return(full_results)
}

#===============================================================================
# Improved Export Functions
#===============================================================================

# Export PCA-transformed dataset directly from stored results
export_pca_components <- function(results, n_components, output_file = NULL) {
  # Extract necessary data from results
  pca_model <- results$pca_results$pca_result
  preserved_cols <- results$preserved_cols
  
  # Get the PC scores directly from the PCA results
  pc_scores <- as.data.frame(pca_model$x)
  
  # Validate n_components
  max_components <- ncol(pc_scores)
  if (n_components > max_components) {
    warning(paste("n_components exceeds available principal components (", max_components, "). Using max available."))
    n_components <- max_components
  }
  
  # Select desired number of components
  pc_selected <- pc_scores[, 1:n_components]
  
  # Add meaningful column names
  colnames(pc_selected) <- paste0("PC", 1:n_components)
  
  # Prepare final dataset with preserved columns
  final_data <- pc_selected
  
  # Add timestamp as first column if it exists
  if (!is.null(preserved_cols$timestamp)) {
    cat("Adding timestamp as first column\n")
    final_data <- cbind(timestamp = preserved_cols$timestamp, final_data)
  }
  
  # Add close price as last column if it exists
  if (!is.null(preserved_cols$close)) {
    cat("Adding close price as last column\n")
    final_data$close <- preserved_cols$close
  }
  
  # Define export path
  if (is.null(output_file)) {
    output_file <- paste0("btcusdc_pca_components_", n_components, ".csv")
  }
  export_path <- file.path(dirname(file_path), output_file)
  
  # Export the dataset
  write_csv(final_data, export_path)
  cat("Successfully exported", n_components, "principal components to:", export_path, "\n")
  
  return(final_data)
}

# Export multiple versions with different component counts
export_multiple_versions <- function(results, component_counts, base_filename = "btcusdc_pca_components") {
  for (n in component_counts) {
    output_file <- paste0(base_filename, "_", n, ".csv")
    export_pca_components(results, n, output_file)
  }
}

#===============================================================================
# Script Execution
#===============================================================================

# Run the full analysis
results <- run_pca_analysis(file_path)

# Export datasets with different numbers of components

# Option 3: 80% variance (approximately 30 components)
export_pca_components(results, n_components = 30, 
                      output_file = "btcusdc_pca_components_30.csv")

# Option 4: 90% variance (approximately 44 components)
export_pca_components(results, n_components = 44, 
                      output_file = "btcusdc_pca_components_44.csv")

# Alternative: Export all versions at once
# export_multiple_versions(results, c(20, 30, 44))
