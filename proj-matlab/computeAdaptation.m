function adaptedMeanMatrix = computeAdaptation(UBM, meanStatisticMatrix,...
                                     weightStatisticList, relevanceFactor)
%computeStatistics Compute the adaptation of means parameter
%
%   Imput: *UBM: A UBM Matlab object
%          *meansStatisticMatrix: A NxM matrix of mean statistics compu-
%           ted on data training from the UBM. N is the number of mixtu-
%           res and M the number of features
%          *adaptatioCoefficientList: A 1xN matrix which stores the
%           adaptation coefficients of each mixture
%
%   Output: A NxM means matrix adapted from UBM, where N is the nuumber
%           of mixtures and M the number of features
%   
%   Speech Processing Project - Computer Engineering CIn/UFPE-Recife BR
%   Sérgio Renan Vieira


%% Computing the Adaptation Coefficients [E1. 14]
adaptationCoefficientList = weightStatisticList./...
                            (weightStatisticList + relevanceFactor);
adaptationCoefficientList(isnan(adaptationCoefficientList)) = 0;
adaptationCoefficientList = repmat(adaptationCoefficientList', 1,...
                                  size(UBM.mu, 2));
                              
%% Computing mean adaptation [Eq. 12]
meanUBMMatrix = UBM.mu;                              
adaptedMeanMatrix = adaptationCoefficientList.*meanStatisticMatrix + ...
                    (1 - adaptationCoefficientList).*meanUBMMatrix;

