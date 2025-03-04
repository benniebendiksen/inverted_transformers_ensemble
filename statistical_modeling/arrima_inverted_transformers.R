# ARIMA Analysis Script for BTC-USDC 4H Data
# Clear environment and load required libraries

install.packages("forecast", dependencies=TRUE)
install.packages("tseries", dependencies=TRUE)


rm(list=ls())
library(TTR)      # For technical indicators
library(forecast) # For ARIMA models
library(tseries)  # For time series analysis
library(car)      # For regression diagnostics
library(lmtest)   # For coefficient testing

# Set options for numeric display
options(digits = 9)  # Set the number of digits to display
options(scipen = 6)  # Make scientific notation less likely

# Import data
mydata <- read.csv("btcusdc_4h_historical.csv", header=TRUE, sep = ",")

# Feature Engineering

# First, calculate log returns for price data
mydata$loggro <- c(NA, diff(log(mydata$close)))

# Calculate log returns for highs and lows
mydata$logHi <- log(mydata$high)
mydata$logLo <- log(mydata$low)
mydata$logHiGro <- c(NA, diff(mydata$logHi))
mydata$logLoGro <- c(NA, diff(mydata$logLo))
mydata$logHiMinusLogLo <- mydata$logHi - mydata$logLo
mydata$logHiGroMinusLogLoGro <- mydata$logHiGro - mydata$logLoGro

# Calculate squared and cubed log returns (for volatility modeling)
mydata$loggro2 <- mydata$loggro^2
mydata$loggro3 <- mydata$loggro^3

# Calculate basic moving averages
mydata$SMA6 <- SMA(mydata$close, n = 6)
mydata$EMA6 <- EMA(mydata$close, n = 6)
mydata$RMA6 <- EMA(mydata$close, n = 6, wilder = TRUE)  # Wilder's RMA

mydata$SMA14 <- SMA(mydata$close, n = 14)
mydata$EMA14 <- EMA(mydata$close, n = 14)
mydata$RMA14 <- EMA(mydata$close, n = 14, wilder = TRUE)

mydata$SMA21 <- SMA(mydata$close, n = 21)
mydata$EMA21 <- EMA(mydata$close, n = 21)
mydata$RMA21 <- EMA(mydata$close, n = 21, wilder = TRUE)

mydata$SMA42 <- SMA(mydata$close, n = 42)
mydata$EMA42 <- EMA(mydata$close, n = 42)
mydata$RMA42 <- EMA(mydata$close, n = 42, wilder = TRUE)

mydata$SMA84 <- SMA(mydata$close, n = 84)
mydata$EMA84 <- EMA(mydata$close, n = 84)
mydata$RMA84 <- EMA(mydata$close, n = 84, wilder = TRUE)

# Calculate relative volume and volatility features
mydata$rvol <- mydata$volume / SMA(mydata$volume, n = 21)
mydata$rvolgro <- mydata$loggro * mydata$SMA21
mydata$loggrorvol2 <- mydata$loggro * mydata$SMA21 * mydata$SMA21
mydata$loggro2rvol <- mydata$loggro * mydata$loggro * mydata$SMA21
mydata$loggro2rvol2 <- mydata$loggrorvol2 * mydata$loggro

# Calculate candlestick pattern features
mydata$body_size <- abs(mydata$close - mydata$open)
mydata$low_wick <- ifelse(mydata$close > mydata$open, mydata$open - mydata$low, mydata$close - mydata$low)
mydata$high_wick <- ifelse(mydata$close < mydata$open, mydata$high - mydata$open, mydata$high - mydata$close)

# Replace body size of zero with a small number to prevent division by zero
mydata$body_size[mydata$body_size == 0] <- 1e-8

# Calculate wick to body ratios
mydata$low_wick_body_ratio <- ifelse(is.na(mydata$low_wick / mydata$body_size), 0, mydata$low_wick / mydata$body_size)
mydata$high_wick_body_ratio <- ifelse(is.na(mydata$high_wick / mydata$body_size), 0, mydata$high_wick / mydata$body_size)

# Calculate moving averages of wick ratios
mydata$lowWickEma14 <- EMA(mydata$low_wick_body_ratio, n = 14)
mydata$highWickEma14 <- EMA(mydata$high_wick_body_ratio, n = 14)
mydata$lowWickEma14rel <- mydata$low_wick_body_ratio
mydata$highWickEma14rel <- (mydata$high_wick_body_ratio) / EMA(mydata$high_wick_body_ratio, n = 14)

# Calculate range-based features
mydata$relRangeSma42 <- SMA(mydata$high - mydata$low, n = 42)
mydata$relBodySma100 <- (mydata$body_size^2 / EMA(mydata$body_size^2, n = 42))

# Calculate combined technical indicators
mydata$lowWickRatioEma42 <- (mydata$rvol^2) / EMA(mydata$low_wick_body_ratio, n = 42)
mydata$highWickRatioEma42 <- (mydata$rvol^2) / SMA(mydata$high_wick_body_ratio, n = 42)
mydata$LRema6 <- mydata$loggro * mydata$EMA6
mydata$LRrma14 <- mydata$loggro * mydata$RMA14
mydata$L2Rrma14 <- mydata$loggro2 * mydata$RMA14
mydata$LR2rma14 <- mydata$loggro2 * mydata$RMA14 * mydata$RMA14

# Use RSI from the dataset
mydata$RSI_smoothed <- EMA(mydata$RSI, n = 3)

# Create a forecast feature for the model (simple prediction based on recent returns)
mydata$forecastBit <- c(rep(NA, 6), SMA(mydata$loggro, n = 6)[-c(1:6)])
mydata$forecastBitNoShift <- SMA(mydata$loggro, n = 6)

# Convert the data to a time series object
# Select relevant columns for modeling
my_ts <- ts(mydata[, c("timestamp", "open", "high", "low", "close", "volume", 
                       "loggro", "loggro2", "loggro3", "SMA6", "EMA6", "RMA6", 
                       "SMA14", "EMA14", "RMA14", "SMA21", "EMA21", "RMA21",
                       "SMA42", "EMA42", "RMA42", "SMA84", "EMA84", "RMA84",
                       "rvol", "rvolgro", "loggrorvol2", "loggro2rvol", "loggro2rvol2",
                       "body_size", "low_wick", "high_wick", "low_wick_body_ratio", "high_wick_body_ratio",
                       "lowWickEma14", "highWickEma14", "lowWickEma14rel", "highWickEma14rel",
                       "relRangeSma42", "relBodySma100", "lowWickRatioEma42", "highWickRatioEma42",
                       "LRema6", "LRrma14", "L2Rrma14", "LR2rma14",
                       "RSI", "RSI_smoothed", "RSI_signal", "forecastBit", "forecastBitNoShift")], 
            start = 1, frequency = 1)

# Create lagged variables for all features to use as predictors
# We'll use a function to create lagged versions for all relevant columns
create_lagged <- function(column_name) {
  lagged_name <- paste0(column_name, "L")
  assign(lagged_name, c(NA, my_ts[1:(length(my_ts[, column_name])-1), column_name]), envir = .GlobalEnv)
  return(lagged_name)
}

# List of columns to create lags for (excluding timestamp and other non-numeric columns)
columns_to_lag <- c("loggro", "loggro2", "loggro3", "rvolgro", "loggrorvol2", "loggro2rvol", 
                    "loggro2rvol2", "lowWickRatioEma42", "highWickRatioEma42", "LRema6", 
                    "LRrma14", "L2Rrma14", "LR2rma14", "RSI", "RSI_smoothed", "forecastBit", 
                    "forecastBitNoShift", "EMA6", "EMA14", "EMA21", "SMA21")

# Create lagged versions of all features
lagged_columns <- lapply(columns_to_lag, create_lagged)

# Prepare data frame for ARIMA modeling
my_df <- data.frame(loggro = my_ts[, "loggro"])

# Add all lagged variables to the data frame
for (col in lagged_columns) {
  my_df[[col]] <- get(col)
}

# Create combined feature sets for ARIMA modeling
# Similar to the original script, we create different combinations of features
hpmFCB <- cbind(forecastBitL, loggroL, loggro2L, loggro3L)
hpmFCBnoShift <- cbind(forecastBitNoShiftL, loggroL, loggro3L)
hpmRSI <- cbind(RSIL, RSI_smoothedL, loggroL, loggro2L)
hpmTech <- cbind(EMA6L, EMA14L, EMA21L, loggroL, loggro2L)
hpmWicks <- cbind(lowWickRatioEma42L, highWickRatioEma42L, loggroL, loggro2L)
hpmCombined <- cbind(forecastBitNoShiftL, RSIL, EMA21L, loggroL, loggro2L)

# Fit ARIMA model
# Use the same order and parameters as the original script
Arima_model <- Arima(my_df$loggro, order = c(2L, 0L, 3L), 
                     xreg = hpmFCBnoShift, 
                     transform.pars = FALSE, 
                     method = c("ML"),
                     optim.control = list(reltol = 1e-10, maxit = 1000))

# Print model summary and coefficient tests
summary(Arima_model)
coeftest(Arima_model)

# Get fitted values from the model
loggrohatr <- fitted(Arima_model)

# Create data frame with forecasts and actual values
forecasts_df <- data.frame(
  "loggrohatr" = loggrohatr, 
  "timestamp" = mydata$timestamp, 
  "close" = mydata$close, 
  "open" = mydata$open, 
  "high" = mydata$high, 
  "low" = mydata$low
)

# Create a trading strategy based on forecasts
forecasts_df$buy <- ifelse(forecasts_df$loggrohatr > 0, 1, 0)

# Calculate portfolio value with compounding
forecasts_df$portfolio <- NA
initial_periods <- min(48, nrow(forecasts_df))
for (j in 1:initial_periods) {
  forecasts_df$portfolio[j] <- 1
}

for(i in (initial_periods+1):nrow(forecasts_df)) {
  forecasts_df$portfolio[i] <- ifelse(!is.na(forecasts_df$portfolio[i-1]),
                                      (1 + forecasts_df$buy[i] * ((forecasts_df$close[i] - forecasts_df$open[i]) / forecasts_df$open[i])) * forecasts_df$portfolio[i-1],
                                      NA)
}

# Add a time index for plotting
forecasts_df$t <- seq_along(forecasts_df$portfolio)

# Plot portfolio value over time
plot(forecasts_df$t, forecasts_df$portfolio, type="l", 
     main="Portfolio Value Over Time",
     xlab="Time", ylab="Portfolio Value",
     col="blue")

# Calculate drawdowns
forecasts_df$peak <- NA
forecasts_df$peak[1] <- 1
for(i in 2:nrow(forecasts_df)) {
  forecasts_df$peak[i] <- ifelse(forecasts_df$portfolio[i] > forecasts_df$peak[i-1], 
                                 forecasts_df$portfolio[i], 
                                 forecasts_df$peak[i-1])
}

forecasts_df$portmin <- ifelse(forecasts_df$buy > 0.5,
                               (forecasts_df$low/forecasts_df$close) * forecasts_df$portfolio,
                               forecasts_df$portfolio)

forecasts_df$drawdown <- (1 - (forecasts_df$portmin / forecasts_df$peak))

# Print maximum drawdown
cat("Maximum drawdown:", max(forecasts_df$drawdown, na.rm = TRUE), "\n")

# Calculate additional performance metrics
total_return <- last(forecasts_df$portfolio) / first(forecasts_df$portfolio[!is.na(forecasts_df$portfolio)]) - 1
annual_return <- (1 + total_return)^(365 / (nrow(forecasts_df) * 4)) - 1  # Assuming 4-hour data
win_rate <- sum(forecasts_df$buy == 1 & forecasts_df$close > forecasts_df$open, na.rm = TRUE) / sum(forecasts_df$buy == 1, na.rm = TRUE)

cat("Total Return:", total_return * 100, "%\n")
cat("Annualized Return:", annual_return * 100, "%\n")
cat("Win Rate:", win_rate * 100, "%\n")

# Try alternate model specifications to compare performance
# Use auto.arima to find the best model parameters
auto_model <- auto.arima(my_df$loggro, xreg = hpmCombined, 
                         ic = "aic", 
                         seasonal = FALSE,
                         approximation = FALSE)

summary(auto_model)
coeftest(auto_model)

# Get fitted values from the auto model
auto_loggrohatr <- fitted(auto_model)

# Compare the performance of both models
cor(loggrohatr[!is.na(loggrohatr)], auto_loggrohatr[!is.na(auto_loggrohatr)])

# Plot the forecasts from both models
plot(my_df$loggro, type="l", col="black", main="Actual vs. Predicted Returns", ylab="Log Return")
lines(loggrohatr, col="blue")
lines(auto_loggrohatr, col="red")
legend("topleft", legend=c("Actual", "Manual ARIMA", "Auto ARIMA"), 
       col=c("black", "blue", "red"), lty=1)

# Export results
write.csv(forecasts_df, "btcusdc_4h_arima_results.csv", row.names = FALSE)