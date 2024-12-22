% Set wordLimit
wordLimit = 2000;

% Folder path
folderPath = 'fomc_minutes_pdfs/';
files = dir(fullfile(folderPath, '*.pdf'));  % Get all .pdf files

% Initialize a table to save the results
results = table('Size', [0 3], 'VariableTypes', {'string', 'string', 'double'}, ...
    'VariableNames', {'FileName', 'Date', 'SentimentScore'});

% Iterate through each file, get sentiment score, and save to the table
for i = 1:length(files)
    filePath = fullfile(folderPath, files(i).name);
    %[dateStr, score] = getSentimentScoreFromFile(filePath, wordLimit);
    [dateStr, score] = getSentimentScoreFromPDF(filePath, wordLimit);
    % Add the result to the table
    newRow = {files(i).name, dateStr, score};
    results = [results; newRow]; %#ok<AGROW>  % Update the table
end

% Display the results table
disp(results);

% Save the table to a CSV file
writetable(results, 'sentiment_scores_pdfs_4o_2.csv');
