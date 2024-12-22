clear;

% Read the sentiment score file sentiment_scores_pdfs_4o.csv
sentiment_data = readtable('sentiment_scores_pdfs_4o.csv');

% Convert the Date column to a string array, then use datenum to convert to dates
sentiment_dates = datenum(string(sentiment_data.Date), 'yyyymmdd'); % Convert to date format
sentiment_scores = sentiment_data.SentimentScore;

% Read the S&P 500 data from sp500_2009_to_present.xlsx
sp500_data = readtable('sp500_2009_to_present.xlsx');

% Extract the date column and ensure it is in string format
sp500_dates = datenum(string(sp500_data.Date), 'yyyy-mm-dd HH:MM:SS'); % Convert to date format
sp500_prices = sp500_data.Price; % Extract closing prices

% Interpolate sentiment scores to the S&P 500 date timeline
fomc_sentiment_interp = interp1(sentiment_dates, sentiment_scores, sp500_dates, 'linear', 'extrap');
%%

% Create the plot
figure;

% Plot S&P 500 prices on the left Y-axis
yyaxis left;
plot(sp500_dates, sp500_prices, '-b', 'DisplayName', 'S&P 500');
ylabel('S&P 500 Price');
hold on;

% Plot interpolated FOMC sentiment scores on the right Y-axis
yyaxis right;
plot(sp500_dates, fomc_sentiment_interp, '-r', 'DisplayName', 'Interpolated FOMC Sentiment');
ylabel('FOMC Sentiment Score');

% Set the title and X-axis label for the plot
xlabel('Date');
title('FOMC Sentiment vs S&P 500');

% Format the date on the X-axis
datetick('x', 'yyyy');
legend('S&P 500', 'FOMC Sentiment');
hold off;
%%

% Remove potential NaN values after interpolation to ensure the lengths of both datasets match
valid_idx = ~isnan(fomc_sentiment_interp) & ~isnan(sp500_prices); % Find valid indices

% Use only valid data for correlation analysis
fomc_sentiment_interp_valid = fomc_sentiment_interp(valid_idx);
sp500_prices_valid = sp500_prices(valid_idx);

% Create a time variable in numeric form (use datenum or custom indices)
time_variable = sp500_dates(valid_idx); % Use date sequence as the time variable
%%

% Build a regression model with sentiment scores and time
mdl_with_sentiment = fitlm([fomc_sentiment_interp_valid, time_variable], sp500_prices_valid);

% Output model results, including p-values and t-values
disp(mdl_with_sentiment);

% Compare R² and adjusted R² (model without sentiment scores)
mdl_without_sentiment = fitlm(time_variable, sp500_prices_valid);
disp(['R² without sentiment: ', num2str(mdl_without_sentiment.Rsquared.Ordinary)]);
disp(['Adjusted R² without sentiment: ', num2str(mdl_without_sentiment.Rsquared.Adjusted)]);
disp(['R² with sentiment: ', num2str(mdl_with_sentiment.Rsquared.Ordinary)]);
disp(['Adjusted R² with sentiment: ', num2str(mdl_with_sentiment.Rsquared.Adjusted)]);

% Compare AIC and BIC values
disp(['AIC without sentiment: ', num2str(mdl_without_sentiment.ModelCriterion.AIC)]);
disp(['BIC without sentiment: ', num2str(mdl_without_sentiment.ModelCriterion.BIC)]);
disp(['AIC with sentiment: ', num2str(mdl_with_sentiment.ModelCriterion.AIC)]);
disp(['BIC with sentiment: ', num2str(mdl_with_sentiment.ModelCriterion.BIC)]);

% Perform stepwise regression to automatically select significant variables
X_full = [fomc_sentiment_interp_valid, time_variable];
mdl_stepwise = stepwiselm(X_full, sp500_prices_valid);
disp(mdl_stepwise);

% Create a predictor matrix with sentiment scores and time variables (including polynomial terms)
X_full = [fomc_sentiment_interp_valid, fomc_sentiment_interp_valid.^2, fomc_sentiment_interp_valid.^3, ...
          time_variable, time_variable.^2, time_variable.^3];

% Perform stepwise regression to select the optimal model
mdl_stepwise = stepwiselm(X_full, sp500_prices_valid);

% Display the selected optimal model
disp(mdl_stepwise);
%%

% Manual K-fold cross-validation
K = 5; % 5-fold cross-validation
n = length(sp500_prices_valid); % Sample size of valid data
indices = crossvalind('Kfold', n, K); % Create cross-validation indices

% Initialize variables to store mean squared errors (MSE)
mse_with_sentiment = zeros(K, 1);
mse_without_sentiment = zeros(K, 1);

for k = 1:K
    % Indices for training and testing sets
    test_idx = (indices == k);
    train_idx = ~test_idx;
    
    % Build models using training data (with sentiment scores and time)
    mdl_with_sentiment = fitlm([fomc_sentiment_interp_valid(train_idx), time_variable(train_idx)], sp500_prices_valid(train_idx));
    
    % Build models using training data (time variable only)
    mdl_without_sentiment = fitlm(time_variable(train_idx), sp500_prices_valid(train_idx));
    
    % Predict on the test set
    predictions_with_sentiment = predict(mdl_with_sentiment, [fomc_sentiment_interp_valid(test_idx), time_variable(test_idx)]);
    predictions_without_sentiment = predict(mdl_without_sentiment, time_variable(test_idx));
    
    % Calculate mean squared error (MSE) on the test set
    mse_with_sentiment(k) = mean((sp500_prices_valid(test_idx) - predictions_with_sentiment).^2);
    mse_without_sentiment(k) = mean((sp500_prices_valid(test_idx) - predictions_without_sentiment).^2);
end

% Calculate average mean squared error (MSE)
avg_mse_with_sentiment = mean(mse_with_sentiment);
avg_mse_without_sentiment = mean(mse_without_sentiment);

% Output cross-validation results
disp(['Average CV MSE with sentiment: ', num2str(avg_mse_with_sentiment)]);
disp(['Average CV MSE without sentiment: ', num2str(avg_mse_without_sentiment)]);

% Extract regression coefficients from the final regression model
mdl_with_sentiment_final = fitlm([fomc_sentiment_interp_valid, time_variable], sp500_prices_valid);
coefficients = mdl_with_sentiment_final.Coefficients.Estimate;
disp(['Sentiment Score Importance (Coefficient): ', num2str(abs(coefficients(2)))]);
disp(['Time Importance (Coefficient): ', num2str(abs(coefficients(3)))]);
