function [sensitivity,specificity] = test_threshold(seiz,nonseiz,thr,cmpr_set)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    if strcmp(cmpr_set,'le')
        TP=sum(seiz<=thr);
        FN=sum(seiz>thr);
        TN=sum(nonseiz>thr);
        FP=sum(nonseiz<=thr);
    elseif strcmp(cmpr_set,'he')
        TP=sum(seiz>=thr);
        FN=sum(seiz<thr);
        TN=sum(nonseiz<thr);
        FP=sum(nonseiz>=thr);
    end
    sensitivity=100*TP/(TP+FN);
    specificity=100*TN/(TN+FP);
end

