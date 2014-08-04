%GMMAdaptation Training GMM speaker models using MAP adaptation
%   
%   Imput: *relevanceFactor: The relevance factor of mean parameter
%          *epsilon: The stable difference between likelihoods
%           after iterations 
%
%   Output: A 1xN cell array of GMM Matlab objects, where N is 
%           the nuumber of speakers computed from UBM adaptation
%
%   Speech Processing Project - Computer Engineering CIn/UFPE-Recife BR
%   Sérgio Renan Vieira

%% loading files
clc
tic
fprintf('Loading Files...\n')
UBM = load('Gender Independent GMM.mat');
GMM = load('Individual Speakers GMM.mat');
utteranceSpeakerList = load('Test Data.mat');
utteranceImposterList = load('Imposter Data.mat');
toc

%% setting variables
UBM = UBM.GMM{3};
GMM = GMM.speakerGMMList;

nSpeakers = size(utteranceSpeakerList.MFCC, 1);
nImposters = size(utteranceImposterList.MFCC, 1);
nUtterances = size(utteranceSpeakerList.MFCC, 2); %same for Sp. and Imp. 
nMixtures = size(UBM.mu, 2); %is equal to GMM mixtures number

UBMMeanList = UBM.mu;
UBMSigmaList = UBM.Sigma;
UBMWeightList = UBM.PComponents;

%% computing the UBM Log Likelihood of Imposters
fprintf('\nComputing the UBM log likelihoods of imposters...\n')
imposterLogLikelihoodUBMMatrix = zeros(nImposters, nUtterances); 
imposterMixtureIdx = cell(nImposters, nUtterances);
imposterFrames = cell(nImposters, nUtterances);

tic
for iImposter = 1:nImposters
    fprintf('   Imposter %d\n', iImposter)
    for jUtterance = 1:nUtterances
        uttteranceImposterFrames =... 
        utteranceImposterList.MFCC(iImposter, jUtterance).config3;
        imposterFrames{iImposter, jUtterance} =...
                                              uttteranceImposterFrames;
        
        probabilities = computeNormalDensity(nMixtures, UBMMeanList,...
                        UBMSigmaList, uttteranceImposterFrames);
        
        [idx, logLikelihood] = computeLogLikelihood(probabilities,...
                                  UBMWeightList, 5);                            
        
        imposterLogLikelihoodUBMMatrix(iImposter, jUtterance) =...
                                                        logLikelihood;
        imposterMixtureIdx{iImposter, jUtterance} = idx;
    end 
end
toc

%% figuring out the evaluation
fprintf('\nFiguring out the evaluation by Speaker\n')
speakerScoreList = cell(nSpeakers);

tic
for iSpeaker = 1:nSpeakers
    fprintf('Speaker %d\n', iSpeaker)
    
    %GMM of the actual speaker
    speakerGMM = GMM{iSpeaker};
    speakerMu = speakerGMM.mu;
    speakerSigma = speakerGMM.Sigma;
    speakerWeight = speakerGMM.PComponents;
    
    %stores all speaker utterances scores
    positiveScoreList = zeros(nUtterances);
    
    fprintf('   Computing the utterances score\n')
    tic
    %computing the speaker utterances score
    for jUtterance = 1:nUtterances
        %computing the UBM densitites
        utteranceFrames =... 
        utteranceSpeakerList.MFCC(iSpeaker, jUtterance).config3;
        
        probabilities = computeNormalDensity(nMixtures, UBMMeanList,...
                         UBMSigmaList, utteranceFrames);
                     
        %computing the UBM log likelihood with C best densities
        [idx, logLikelihoodUBM] = computeLogLikelihood(probabilities,...
                                  UBMWeightList, 5);
        
        %computing the speaker utterance densities with the C best
        %mixtures
        probabilities = computeCDensities(utteranceFrames, speakerMu,...
                                           speakerSigma, idx);
        
        %computing the speaker utterance log likelihood
        weightList = speakerWeight(idx);
        likelihoodList = sum(weightList.*probabilities, 2);
        logLikelihoodGMM = sum(log(likelihoodList));
        
        %storing the speaker utterance score
        positiveScoreList(jUtterance) = logLikelihoodGMM -...
                                        logLikelihoodUBM;                                 
    end
    toc
    
    fprintf('   Computing the imposters GMM scores\n\n')
    tic
    imposterLogLikelihoodMatrix = zeros(nImposters, nUtterances);
    %computing the imposter log likelihood for the GMM of the actual
    %speaker
    for iImposter = 1:nImposters
        for jUtterance = 1:nUtterances
            imposterFrameList = imposterFrames{iImposter, jUtterance};
            idx = imposterMixtureIdx{iImposter, jUtterance};
            
            %computing the imposter utterance densities with the C best
            %mixtures
            probabilities = computeCDensities(imposterFrameList,...
                                      speakerMu, speakerSigma, idx);
            
            %cmputing the imposter utterance log likelihood
            weightList = speakerWeight(idx);
            likelihoodList = sum(weightList.*probabilities, 2);
            logLikelihoodImposter = sum(log(likelihoodList));
            
            imposterLogLikelihoodMatrix(iImposter, jUtterance) =...
                logLikelihoodImposter; 
        end 
    end
    toc
    %computing the imposter scores, by imposter and utterance
    imposterScoreMatrix = imposterLogLikelihoodMatrix -...
                          imposterLogLikelihoodUBMMatrix; 
    %storing the postive (from GMM speaker) and negative 
    %(from imposters) scores
    score.positive = positiveScoreList;
    score.negative = imposterScoreMatrix;
    speakerScoreList{iSpeaker} = score;
end
toc

save('Speakers Scores.mat', 'speakerScoreList');
