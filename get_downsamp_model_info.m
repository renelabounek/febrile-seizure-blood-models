function M = get_downsamp_model_info(m,Mfull,niter)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    M(1,1).inmodel = reshape([m(:,1).inmodel]',size(m(1,1).inmodel,2),niter)';
    M(1,1).inmodel_unique = unique(M.inmodel,'rows');
    M(1,1).inmodel_unique_prob = zeros(size(M.inmodel_unique,1),1);

    for res = 1:size(M.inmodel_unique,1)
        for iter = 1:niter
            M.inmodel_unique_prob(res,1) = M.inmodel_unique_prob(res,1) + ...
                double(sum(M.inmodel_unique(res,:) == M.inmodel(iter,:)) == size(m(1,1).inmodel,2));
        end
    end
    M(1,1).inmodel_unique_prob = 100*M.inmodel_unique_prob/niter;
    M(1,1).best = find(M.inmodel_unique_prob==max(M.inmodel_unique_prob));
    M(1,1).match = find(sum(Mfull.inmodel_unique(Mfull.best,:)==M.inmodel_unique,2)==size(Mfull.inmodel,2)==1);
    if ~isempty(M(1,1).match)
        M(1,1).pos = sum(M.inmodel == M.inmodel_unique(M.match,:),2) == size(M.inmodel,2);
        M(1,1).varid = find(M.inmodel_unique(M.match,:)==1);
        b = [m.b]';
        b = b(M.pos,M.varid);
        M.b(:,1) =  mean(b);
        M.b(:,2) =  std(b);
        pval = [m.pval]';
        pval = pval(M.pos,M.varid);
        M.pval = median(pval);
        r = [m(M.pos,1).r]';
        M.r = [mean(r) std(r)];
        M.p_r = median([m(M.pos,1).p_r]);
        R2 = [m(M.pos,1).R2]';
        M.R2 = [mean(R2) std(R2)];

        M.prob = M.inmodel_unique_prob(M.match);
        M.unique_models = size(M.inmodel_unique,1);
        M.matchisbest = M.match == M.best;
        
        th = [m(M.pos,1).threshold]';
        M.threshold = [mean(th) std(th)];
        
        th = [m(M.pos,1).sensitivity]';
        M.sensitivity = [mean(th) std(th)];
        th = [m(M.pos,1).specificity]';
        M.specificity = [mean(th) std(th)];
        
        th = [m(M.pos,1).testSE]';
        M.testSE = [mean(th) std(th)];
        th = [m(M.pos,1).testSP]';
        M.testSP= [mean(th) std(th)];
        
        th = arrayfun(@(x) x.stats.rmse,m); th = th(M.pos,1);
        M.rmse = [mean(th) std(th)];
        th = arrayfun(@(x) x.stats.fstat,m); th = th(M.pos,1);
        M.fstat = [mean(th) std(th)];
    end
end

