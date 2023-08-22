function m = regress_undersampled_model2(XX0f,XX0m,XX1f,XX1m,...
    sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,...
    sizeXX0f_downsamp,sizeXX0m_downsamp,sizeXX1f_downsamp,sizeXX1m_downsamp,...
    adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    
    m = struct([]);
    
    permXX0f_downsamp = randperm(sizeXX0f); permXX0f_downsamp = sort(permXX0f_downsamp(1:sizeXX0f_downsamp))';
    permXX0m_downsamp = randperm(sizeXX0m); permXX0m_downsamp = sort(permXX0m_downsamp(1:sizeXX0m_downsamp))';
    permXX1f_downsamp = randperm(sizeXX1f); permXX1f_downsamp = sort(permXX1f_downsamp(1:sizeXX1f_downsamp))';
    permXX1m_downsamp = randperm(sizeXX1m); permXX1m_downsamp = sort(permXX1m_downsamp(1:sizeXX1m_downsamp))';
    
    adasyn_features_downsamp                = [XX1f(permXX1f_downsamp,1:end-1); XX1m(permXX1m_downsamp,1:end-1)];
    adasyn_labels_downsamp                   = [true([size(permXX1f_downsamp,1) 1]);  false([size(permXX1m_downsamp,1) 1])  ];
    
    [adasyn_featuresSyn, adasyn_labelsSyn] = ADASYN(adasyn_features_downsamp, adasyn_labels_downsamp, adasyn_beta, adasyn_kDensity, adasyn_kSMOTE, adasyn_featuresAreNormalized);
    
    XX_downsamp0 = [XX0f(permXX0f_downsamp,:); XX0m(permXX0m_downsamp,:); XX1f(permXX1f_downsamp,:); XX1m(permXX1m_downsamp,:)];
    XX_downsamp = [XX_downsamp0; [adasyn_featuresSyn, adasyn_labelsSyn] ];
    YY_downsamp0 = [zeros(sizeXX0f_downsamp+sizeXX0m_downsamp,1); ones(sizeXX1f_downsamp+sizeXX1m_downsamp,1)];
    YY_downsamp = [YY_downsamp0; ones(size(adasyn_labelsSyn))];
    
    [m(1,1).b,m(1,1).se,m(1,1).pval,m(1,1).inmodel,m(1,1).stats,m(1,1).nextstep,m(1,1).history] = stepwisefit(XX_downsamp,YY_downsamp,'penter',0.05);
    Y_predict = sum( repmat(m(1,1).b(m(1,1).inmodel==1)',size(XX_downsamp0,1),1).*XX_downsamp0(:,m(1,1).inmodel==1) ,2); % Predicted signal Y_predict
    [r, pr]=corrcoef(YY_downsamp0,Y_predict);
    m(1,1).Y_predict = Y_predict;
    m(1,1).r = r(1,2);
    m(1,1).p_r = pr(1,2);
    m(1,1).R2 = (1 - m(1,1).stats(1,1).SSresid / m(1,1).stats(1,1).SStotal)*100;
    tmp=estimate_threshold(Y_predict(YY_downsamp0==1),Y_predict(YY_downsamp0==0),cell(1,1),998,'he');
    m(1,1).threshold = tmp.OptimThr;
    m(1,1).sensitivity = tmp.OptimSensitivity;
    m(1,1).specificity = tmp.OptimSpecificity;
    
    testXX0f = identify_samples_outside_permut(sizeXX0f,permXX0f_downsamp);
    testXX0m = identify_samples_outside_permut(sizeXX0m,permXX0m_downsamp);
    testXX1f = identify_samples_outside_permut(sizeXX1f,permXX1f_downsamp);
    testXX1m = identify_samples_outside_permut(sizeXX1m,permXX1m_downsamp);
    
    XXtest = [XX0f(testXX0f,:); XX0m(testXX0m,:); XX1f(testXX1f,:); XX1m(testXX1m,:) ];
    YYtest = [zeros(size(testXX0f,1)+size(testXX0m,1),1); ones(size(testXX1f,1)+size(testXX1m,1),1)];
    Y_predict_test = sum( repmat(m(1,1).b(m(1,1).inmodel==1)',size(XXtest,1),1).*XXtest(:,m(1,1).inmodel==1) ,2); % Predicted signal Y_predict
    
    [m(1,1).testSE,m(1,1).testSP] = test_threshold(Y_predict_test(YYtest==1),Y_predict_test(YYtest==0),m(1,1).threshold,'he');
end

