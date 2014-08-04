%% Separando os dados nas 3 configurações
clc
tic
train = load('Training Data.mat');
toc

data1 = []; data2 = []; data3 = [];

x = 0;
tic
for i = 1:(48*54)
    y = mod(i, 54);
    
    if (y == 0)
        y = 54;
    elseif (y == 1)
        x = x + 1;
    end
    
    data1 = [data1; train.MFCC(x, y).config1];
    data2 = [data2; train.MFCC(x, y).config2];
    data3 = [data3; train.MFCC(x, y).config3];
    
    if (i == (22*54))
        femaleSize = size(data1, 1);
    end    
end
toc

%% Treinando os modelos UBM separados por Gênero e Configuração
fprintf('Treinando female 1...\n')
tic
femaleGMM1 = trainGMM_hnbp(data1(1:femaleSize, :), 256, 3, 1);
toc

fprintf('Treinando male 1...\n')
tic
maleGMM1 = trainGMM_hnbp(data1((femaleSize + 1):end, :), 256, 3, 1);
toc

fprintf('Treinando female 2...\n')
tic
femaleGMM2 = trainGMM_hnbp(data2(1:femaleSize, :), 256, 3, 1);
toc

fprintf('Treinando male 2...\n')
tic
maleGMM2 = trainGMM_hnbp(data2((femaleSize + 1):end, :),...
                          256, 3, 1);
toc

fprintf('Treinando female 3...\n')
tic
femaleGMM3 = trainGMM_hnbp(data3(1:femaleSize, :), 256, 3, 1);
toc

fprintf('Treinando male 3...\n')
tic
maleGMM3 = trainGMM_hnbp(data3((femaleSize + 1):end, :),...
                        256, 3, 1);
toc

%% Reunindo os modelos UBM e salvando 
tic
GMM = {femaleGMM maleGMM1 femaleGMM2 maleGMM2 femaleGMM3 maleGMM3};
toc

fprintf('\n\nSalvando os GMMs...\n')
tic
save ('GMM.mat', 'GMM');
toc