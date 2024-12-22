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

% Remove potential NaN values after interpolation to ensure the lengths of both datasets match
valid_idx = ~isnan(fomc_sentiment_interp) & ~isnan(sp500_prices); % Find valid indices

% Use only valid data for correlation analysis
fomc_sentiment_interp_valid = fomc_sentiment_interp(valid_idx);
sp500_prices_valid = sp500_prices(valid_idx);

% Output their lengths to check if the row counts are consistent
%disp(['FOMC Sentiment Interpolated Valid Length: ', num2str(length(fomc_sentiment_interp_valid))]);
%disp(['S&P 500 Prices Valid Length: ', num2str(length(sp500_prices_valid))]);

% Output the row counts of the full data for further inspection
%disp(['FOMC Sentiment Interpolated Total Length: ', num2str(length(fomc_sentiment_interp))]);
%disp(['S&P 500 Prices Total Length: ', num2str(length(sp500_prices))]);

% Create a time variable in numeric form (use datenum or custom indices)
time_variable = sp500_dates; % Use the date sequence as the time variable
%%

% Create predictor matrices with time and sentiment scores
X_time_and_sentiment = [fomc_sentiment_interp_valid, time_variable]; % Use sentiment scores and time
X_time_only = time_variable; % Use only time as the predictor

% Multiple linear regression (time + sentiment scores)
mdl_time_and_sentiment = fitlm(X_time_and_sentiment, sp500_prices_valid);

% Regression with only the time variable
mdl_time_only = fitlm(X_time_only, sp500_prices_valid);

% Predict S&P 500 prices
predicted_prices_time_and_sentiment = predict(mdl_time_and_sentiment, X_time_and_sentiment);
predicted_prices_time_only = predict(mdl_time_only, X_time_only);

% Plot actual prices and regression curves for both models
figure;

% Plot actual prices
plot(sp500_dates, sp500_prices_valid, 'b', 'DisplayName', 'Actual S&P 500 Prices');
hold on;

% Plot predicted prices from the model using time and sentiment scores
plot(sp500_dates, predicted_prices_time_and_sentiment, 'r--', 'DisplayName', 'Predicted (Time + Sentiment)');

% Plot predicted prices from the model using only time
plot(sp500_dates, predicted_prices_time_only, 'g--', 'DisplayName', 'Predicted (Time Only)');

% Set plot properties
xlabel('Date');
ylabel('S&P 500 Price');
title('Actual vs Predicted S&P 500 Prices');
legend('Location', 'Best');
datetick('x', 'yyyy'); % Format the date as years
hold off;
%%

% Create a predictor matrix with sentiment scores and time variables (including polynomial terms)
X_full = [fomc_sentiment_interp_valid, fomc_sentiment_interp_valid.^2, fomc_sentiment_interp_valid.^3, ...
          time_variable, time_variable.^2, time_variable.^3];

% Perform stepwise regression to select the optimal model
mdl_stepwise = stepwiselm(X_full, sp500_prices_valid);

% Display the selected optimal model
disp(mdl_stepwise);
