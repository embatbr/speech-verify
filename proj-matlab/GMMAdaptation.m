function speakerGMMList = GMMAdaptation(relevanceFactor, epsilon,...
                                        nMaxIterations)
%GMMAdaptation Training GMM speaker models using MAP adaptation
%   
%   Imput: *relevanceFactor: The relevance factor of mean parameter
%          *epsilon: The stable difference between likelihoods
%           after iterations 
%
%   Output: A 1xN cell array of GMM Matlab objects computed from UBM
%           adaptation, where N is the nuumber of speakers
%
%   Speech Processing Project - Computer Engineering CIn/UFPE-Recife BR
%   Sérgio Renan Vieira

clc;

speakerList = load('Training Data.mat');
UBM = load('Gender Independent GMM.mat');

speakerGMMList = cell(1, size(speakerList.MFCC, 1));
for iSpeaker = 1:size(speakerList.MFCC, 1)
    %% join all utterances of a speaker together
    fprintf('Join all utterances of speaker %d together\n', iSpeaker);
    speaker = [];
    for jUtterance = 1:size(speakerList.MFCC, 2)
        speaker = [speaker;...
                   speakerList.MFCC(iSpeaker, jUtterance).config3];       
    end
    
    %% carrying out the adaptation
    fprintf('    Carrying out the adaptation\n');
    %UBM for the third features configuration (MFCC + D1 + D2)
    GMM = UBM.GMM{3};
    
    counter = 0;
    condition = true;
    tic
    while condition
        counter  = counter + 1;
        [weightStatisticList, meanStatisticMatrix] =...
                                        computeStatistics(GMM, speaker);
        adaptedMeanMatrix = computeAdaptation(GMM,...
            meanStatisticMatrix, weightStatisticList, relevanceFactor);
        
        %actual likelihood
        likelihood = mean(log(GMM.pdf(speaker)));
        
        %modifying the speaker GMM from the mean adaptation
        GMM = gmdistribution(adaptedMeanMatrix, GMM.Sigma,...
                            GMM.PComponents);
        
        %new likelihood
        newLikelihood = mean(log(GMM.pdf(speaker)));
        
        %figuring out the actual state of the loop
        likelihoodRatio = abs((newLikelihood - likelihood)/likelihood);
        condition = (likelihoodRatio >= epsilon) &...
                    (counter < nMaxIterations);
    end
    toc
    fprintf('Likelihood Ratio = %f\n', likelihoodRatio);
    fprintf('Iterations: %d\n\n', counter);
%%
    speakerGMMList{iSpeaker} = GMM;
end

%% saving the GMMs
save('Individual Speakers GMM', 'speakerGMMList');
%%
