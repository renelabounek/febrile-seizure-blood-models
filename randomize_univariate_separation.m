function [UU, U] = randomize_univariate_separation(X,grp,variable_name,varID,cmpr_set,female,niter_adasyn)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    
    dataset_size = 0.9:-0.1:0.5;

    XX0f = X(grp==2 & female==1,1);
    XX0m = X(grp==2 & female==0,1);
    XX1f = X((grp==3 | grp==5 | grp==4) & female==1,1);
    XX1m = X((grp==3 | grp==5 | grp==4) & female==0,1);

    sizeXX0f = size(XX0f,1);
    sizeXX0m = size(XX0m,1);
    sizeXX1f = size(XX1f,1);
    sizeXX1m = size(XX1m,1);
    
%     U = struct([]);

  
    for ind = 1:size(dataset_size,2)
        for xnd = 1:niter_adasyn 

            sizeXX0f_downsamp = round(dataset_size(ind)*sizeXX0f);
            sizeXX0m_downsamp = round(dataset_size(ind)*sizeXX0m);
            sizeXX1f_downsamp = round(dataset_size(ind)*sizeXX1f);
            sizeXX1m_downsamp = round(dataset_size(ind)*sizeXX1m);

            permXX0f_downsamp = randperm(sizeXX0f); permXX0f_downsamp = sort(permXX0f_downsamp(1:sizeXX0f_downsamp))';
            permXX0m_downsamp = randperm(sizeXX0m); permXX0m_downsamp = sort(permXX0m_downsamp(1:sizeXX0m_downsamp))';
            permXX1f_downsamp = randperm(sizeXX1f); permXX1f_downsamp = sort(permXX1f_downsamp(1:sizeXX1f_downsamp))';
            permXX1m_downsamp = randperm(sizeXX1m); permXX1m_downsamp = sort(permXX1m_downsamp(1:sizeXX1m_downsamp))';

            XXtrain = [XX0f(permXX0f_downsamp,1); XX0m(permXX0m_downsamp,1); XX1f(permXX1f_downsamp,1); XX1m(permXX1m_downsamp,1)];
            YYtrain = [zeros(sizeXX0f_downsamp+sizeXX0m_downsamp,1); ones(sizeXX1f_downsamp+sizeXX1m_downsamp,1)];

            tmp=estimate_threshold(XXtrain(YYtrain==1),XXtrain(YYtrain==0),variable_name,varID,cmpr_set);
            if xnd == 1 && ind == 1
                U.dataset_size = dataset_size(ind);
                U.threshold = tmp.OptimThr;
                U.sensitivity = tmp.OptimSensitivity;
                U.specificity = tmp.OptimSpecificity;
            else
                U(xnd,ind).dataset_size = dataset_size(ind);
                U(xnd,ind).threshold = tmp.OptimThr;
                U(xnd,ind).sensitivity = tmp.OptimSensitivity;
                U(xnd,ind).specificity = tmp.OptimSpecificity;
            end

            testXX0f = identify_samples_outside_permut(sizeXX0f,permXX0f_downsamp);
            testXX0m = identify_samples_outside_permut(sizeXX0m,permXX0m_downsamp);
            testXX1f = identify_samples_outside_permut(sizeXX1f,permXX1f_downsamp);
            testXX1m = identify_samples_outside_permut(sizeXX1m,permXX1m_downsamp);

            XXtest = [XX0f(testXX0f,:); XX0m(testXX0m,:); XX1f(testXX1f,:); XX1m(testXX1m,:) ];
            YYtest = [zeros(size(testXX0f,1)+size(testXX0m,1),1); ones(size(testXX1f,1)+size(testXX1m,1),1)];
            if xnd == 1 && ind == 1
                [U.testSE,U.testSP] = test_threshold(XXtest(YYtest==1),XXtest(YYtest==0),U.threshold,cmpr_set);     
                U.variable_name = variable_name{1,varID};
                U.varID = varID;
            else
                [U(xnd,ind).testSE,U(1,ind).testSP] = test_threshold(XXtest(YYtest==1),XXtest(YYtest==0),U(1,ind).threshold,cmpr_set);     
                U(xnd,ind).variable_name = variable_name{1,varID};
                U(xnd,ind).varID = varID;
            end
        end
        threshold = [U(:,ind).threshold]';
        sensitivity = [U(:,ind).sensitivity]';
        specificity = [U(:,ind).specificity]';
        testSE = [U(:,ind).testSE]';
        testSP = [U(:,ind).testSP]';
        
        if ind == 1
            UU.dataset_size = dataset_size(ind);
            UU.threshold = [mean(threshold) std(threshold)];
            UU.sensitivity = [mean(sensitivity) std(sensitivity)];
            UU.specificity = [mean(specificity) std(specificity)];
            UU.testSE = [mean(testSE) std(testSE)];
            UU.testSP = [mean(testSP) std(testSP)];
            UU.variable_name = variable_name{1,varID};
            UU.varID = varID;
        else
            UU(1,ind).dataset_size = dataset_size(ind);
            UU(1,ind).threshold = [mean(threshold) std(threshold)];
            UU(1,ind).sensitivity = [mean(sensitivity) std(sensitivity)];
            UU(1,ind).specificity = [mean(specificity) std(specificity)];
            UU(1,ind).testSE = [mean(testSE) std(testSE)];
            UU(1,ind).testSP = [mean(testSP) std(testSP)];
            UU(1,ind).variable_name = variable_name{1,varID};
            UU(1,ind).varID = varID;
        end
    end
end

