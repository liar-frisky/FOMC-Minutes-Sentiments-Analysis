function [dateStr, finalScore] = getSentimentScoreFromPDF(filename, wordLimit)
    % Load environment variables and paths to ensure the API can be used correctly
    loadenv(".env");
    addpath('../..');

    % Set up the model and create the chat object
    modelName = "gpt-4o";
    chat = openAIChat("You are an expert in analyzing economic texts. I will provide text in chunks. After I send all content, I will say '<END>'. Before that, every message will end with '<CONTINUE>'. After '<END>', please provide ONLY the sentiment score (a decimal number between -1 and 1 with up to 5 decimal places). The sentiment score needs to be combined with the economic conditions and market sentiment presented in the text to reflect the prediction of the future market. Do not provide any additional explanation.", ModelName=modelName);
    
    % Extract text content from the PDF file
    try
        textContent = extractFileText(filename);
    catch
        error('Unable to extract text from file: %s', filename);
    end

    % Extract the date from the file name (assuming the format is fomcminutesYYYYMMDD.pdf)
    [~, name, ~] = fileparts(filename);
    dateStr = regexp(name, '\d{8}', 'match', 'once');  % Extract the date in YYYYMMDD format

    % Tokenize the text content
    tokens = split(textContent);  % Split the text into words by whitespace
    numWords = numel(tokens);     % Get the total number of words

    numBatches = ceil(numWords / wordLimit);  % Calculate the number of batches
    messages = messageHistory;  % Initialize the conversation history

    % Send file content in batches
    for i = 1:numBatches
        % Get the current batch of text
        startIdx = (i - 1) * wordLimit + 1;
        endIdx = min(i * wordLimit, numWords);
        batchContent = strjoin(tokens(startIdx:endIdx), ' ');  % Join words into a complete string

        % If not the last batch, add "<CONTINUE>" to indicate more content to come
        if i < numBatches
            batchContent = batchContent + " <CONTINUE>";
        else
            batchContent = batchContent + " <END>";  % Add "<END>" for the last batch
        end

        % Add the message and call GPT
        messages = addUserMessage(messages, batchContent);  % Add content to message history
        [~, response] = generate(chat, messages);  % GPT does not respond mid-process
        pause(4);
    end

    % After sending the last batch, GPT returns the sentiment score
    [text, response] = generate(chat, messages);  % Generate the final score
    pause(4);

    % Use a regular expression to extract the numeric score from the response
    scorePattern = '-?\d+(\.\d+)?';  % Match positive or negative floating-point numbers
    scoreMatch = regexp(text, scorePattern, 'match');

    if isempty(scoreMatch)
        error("Unable to parse a valid score from GPT response: %s", text);
    else
        finalScore = str2double(scoreMatch{1});  % Extract the first matching value
    end

    fprintf('File %s with date %s has a score of: %.5f\n', filename, dateStr, finalScore);
end
