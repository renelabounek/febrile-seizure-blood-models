function s = estimate_threshold(seiz,nonseiz,variable_name,varID,cmpr_set)
%ESTIMATE_THRESHOLD Summary of this function goes here
%   Detailed explanation goes here
    patsall = [seiz; nonseiz];
    perc = 0.01:0.01:1;
    s = struct([]);
    if varID == 997
        s(1,1).name = 'model1';
    elseif varID == 998
        s(1,1).name = 'model2';
    elseif varID == 999
        s(1,1).name = 'model3';
    else
        s(1,1).name = variable_name{1,varID};
    end
    for percx = 1:size(perc,2)
        perc_thr = quantile(patsall, perc(percx));
        if strcmp(cmpr_set,'le')
            TP=sum(seiz<=perc_thr);
            FN=sum(seiz>perc_thr);
            TN=sum(nonseiz>perc_thr);
            FP=sum(nonseiz<=perc_thr);
        elseif strcmp(cmpr_set,'he')
            TP=sum(seiz>=perc_thr);
            FN=sum(seiz<perc_thr);
            TN=sum(nonseiz<perc_thr);
            FP=sum(nonseiz>=perc_thr);
        end
        sensitivity=100*TP/(TP+FN);
        specificity=100*TN/(TN+FP);

        s(1,1).thr(percx,1) = perc_thr;
        s(1,1).TP(percx,1) = TP;
        s(1,1).FN(percx,1) = FN;
        s(1,1).TN(percx,1) = TN;
        s(1,1).FP(percx,1) = FP;
        s(1,1).sensitivity(percx,1) = sensitivity;
        s(1,1).specificity(percx,1) = specificity;
        s(1,1).SensSpecSum(percx,1) = 1.00*sensitivity+specificity;     
    end
    s(1,1).OptimThrPos = find(s(1,1).SensSpecSum==max(s(1,1).SensSpecSum),1,'first');
    s(1,1).OptimThr = s(1,1).thr(s(1,1).OptimThrPos,1);
    s(1,1).OptimSensitivity = s(1,1).sensitivity(s(1,1).OptimThrPos,1);
    s(1,1).OptimSpecificity = s(1,1).specificity(s(1,1).OptimThrPos,1);
    s(1,1).min = min(patsall);
    s(1,1).max = max(patsall);
end

