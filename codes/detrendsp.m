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

% Detrend the S&P 500 data
sp500_detrended_prices = detrend(sp500_prices);

% Interpolate sentiment scores to the S&P 500 date timeline
fomc_sentiment_interp = interp1(sentiment_dates, sentiment_scores, sp500_dates, 'linear', 'extrap');
%%

% Create the plot
figure;

% Plot detrended S&P 500 prices on the left Y-axis
yyaxis left;
plot(sp500_dates, sp500_detrended_prices, '-b', 'DisplayName', 'Detrended S&P 500');
ylabel('Detrended S&P 500 Price');
hold on;

% Plot interpolated FOMC sentiment scores on the right Y-axis
yyaxis right;
plot(sp500_dates, fomc_sentiment_interp, '-r', 'DisplayName', 'Interpolated FOMC Sentiment');
ylabel('FOMC Sentiment Score');

% Set the title and X-axis label for the plot
xlabel('Date');
title('FOMC Sentiment vs Detrended S&P 500');

% Format the date on the X-axis
datetick('x', 'yyyy');
legend('Detrended S&P 500', 'FOMC Sentiment');
hold off;
%%

% Assuming sentiment_score is the sentiment score data and sp500_detrended is the detrended S&P 500 data
% and that these two variables are aligned in the date range

% Combine the two variables into a matrix
data = [fomc_sentiment_interp, sp500_detrended_prices];

% Perform the Engle-Granger cointegration test
[h, pValue, stat, cValue, reg] = egcitest(data);

% Output the test results
if h == 0
    disp('Cannot reject the null hypothesis of no cointegration (no cointegration relationship exists).');
else
    disp('Reject the null hypothesis; a cointegration relationship exists.');
end

% Output the p-value and test statistics
disp(['p-value: ', num2str(pValue)]);
disp(['Test statistic: ', num2str(stat)]);
disp(['Critical value: ', num2str(cValue)]);
