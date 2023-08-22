%FS_MODELS script estimates univariate and multivariate separating properties of
%           pediatric patients with febrile seizures (FS) from febrile and
%           afebrile controls.
%           The script designs three different linear mixture models which
%           improves sensitivity and specificity of the separation.
%
% REQUIRED INPUTS TO BE SET PROPERLY:
%           save_path ... folder where Result figures will be stored
%
%           xls_file ... Excel sheet with the input data
%                        The example sheet is available in the same folder as this function is stored.
%
% =====================================================================================================================================================================
% Please, cite as:
% 
% Papez Jan, Labounek Rene, Jabandziev Petr, Ceska Katarina, Slaba Katerina, Oslejskova Hana, Aulicka Stefania, Nestrasil Igor. Predictive multivariate linear
% mixture models of febrile seizure risk and recurrency: a prospective case-control study. Scientific Reports (2023) [Under Review] 
%
% =====================================================================================================================================================================
% Copyright 2020-2023 Rene Labounek (1,*), Igor Nestrasil (1)
%
% (1) Division of clinical Behavioral Neuroscience, Department of Pediatrics, Masonic Institute for the Developing Brain, University of Minnesota, Minneapolis, MN, USA
% (*) email: rlaboune@umn.edu
%
% febrile-seizure-blood-models is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or any later version.
%
% febrile-seizure-blood-models is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
% or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License along with 
% febrile-seizure-blood-models. If not, see https://www.gnu.org/licenses/.
% =====================================================================================================================================================================

%% Initialize
clear all; close all; clc;
%make results reproducible by resetting the random number generator:
rng('default');
%% Define inputs and fixed variables
save_path='~/figures/fs-results'; %folder where Result figures will be stored.
xls_file = 'febrile_seizures.xlsx'; % Source xlsx sheet with all available data

fntSiz=11; % Font size
thr_p_uncorr = 0.05; % uncorrected critical p-value; normalized at FWE corrected p-value within the code
fig6xmax=1.4;
fig6ymax=0.9;
fig6xmin_zoom=0.48;


[num, txt, raw] = xlsread(xls_file);
cols = size(raw,2);

idx=1;

niter_adasyn = 5000;
m1=struct([]);
m2=struct([]);
m3=struct([]);
M1=struct([]);
M2=struct([]);
M3=struct([]);

m190=struct([]);
m180=struct([]);
m170=struct([]);
m160=struct([]);
m150=struct([]);
m290=struct([]);
m280=struct([]);
m270=struct([]);
m260=struct([]);
m250=struct([]);
m390=struct([]);
m380=struct([]);
m370=struct([]);
m360=struct([]);
m350=struct([]);
%% Analysis variables and value extraction
variable_name = {'GA' 'Age' 'Height' 'Weight' 'HGB' 'Fe' 'Fer' 'TF' 'satFe' 'UIBC'};
variable_pos = [8 10 4 6 14 18:21];

% Another possible script settings. Some script parts does not have to be
% 100% functional. e.g. posdata and postbl variables might need to be adjusted.
%
% variable_name = {'GA' 'Age' 'Height' 'Weight' 'BMI' 'HGB' 'MCV' 'RDW' 'Fe' 'Fer' 'TF' 'satFe' 'TIBC' 'UIBC'};
% variable_pos = [8 10 4 6:7 14 15 17 18:21];
% 
% variable_name = {'GA' 'Age' 'Height' 'Weight' 'BMI' 'RBC' 'HGB' 'Fe' 'Fer' 'TF' 'satFe' 'TIBC' 'UIBC'};
% variable_pos = [8 10 4 6:7 13:14 18:21];

FePos=find(strcmp(variable_name,'Fe')==1);
satFePos=find(strcmp(variable_name,'satFe')==1);

grp=[raw{3:end,11}]';
grp(grp==0) = 9;
grp(grp==1) = 6;
grp(grp==4) = 5;
grp(grp==3) = 1;
grp(grp==6) = 4;
grp(grp==9) = 3;
grp = categorical(grp);

female=strcmp(raw(3:end,strcmp(raw(1,:),'Sex')),'F');

data=zeros(size(raw,1)-2,size(variable_pos,2));
AtNum=zeros(size(raw,1)-2,1);
AtOrder=zeros(size(raw,1)-2,1);
age1stfs=zeros(size(raw,1)-2,1);
patid=zeros(size(raw,1)-2,1);
firstrecord=zeros(size(raw,1)-2,1);
rbc=zeros(size(raw,1)-2,1);
temperature=zeros(size(raw,1)-2,1);
mch=zeros(size(raw,1)-2,1);
vitD=zeros(size(raw,1)-2,1);
sodium=zeros(size(raw,1)-2,1);
seize_duration=zeros(size(raw,1)-2,1);
for ind=1:size(raw,1)-2
        for var = 1:size(variable_pos,2)
                data(ind,var)=tab_translate(raw{ind+2,variable_pos(1,var)});
        end
        % Read another variables from the input xlsx table
        AtNum(ind,1)=tab_translate(raw{ind+2,strcmp(raw(1,:),'FS count')}); % total number of febrile seizure (FS; At = attack)
        AtOrder(ind,1)=tab_translate(raw{ind+2,strcmp(raw(1,:),'attack order')}); % Order of the given FS (At - attack)
        age1stfs(ind,1) = tab_translate(raw{ind+2,strcmp(raw(1,:),'Age 1st FS')}); % age of the 1st FS
        patid(ind,1) = tab_translate(raw{ind+2,strcmp(raw(1,:),'PatID')}); % Patient's ID
        firstrecord(ind,1) = tab_translate(raw{ind+2,strcmp(raw(1,:),'First record')}); % First record (binary)
        rbc(ind,1) = tab_translate(raw{ind+2,strcmp(raw(1,:),'Ery')}); % RBC values
        temperature(ind,1) = tab_translate(raw{ind+2,strcmp(raw(1,:),'BT °C')}); % temperature values
        mch(ind,1) = tab_translate(raw{ind+2,strcmp(raw(1,:),'MCH')}); % MCH values
        vitD(ind,1) = tab_translate(raw{ind+2,strcmp(raw(1,:),'vit.D')}); % vitamin D values
        sodium(ind,1) = tab_translate(raw{ind+2,strcmp(raw(1,:),'sodium')}); % sodium values
        seize_duration(ind,1) = tab_translate(raw{ind+2,strcmp(raw(1,:),'FS duration')}); % Seizure duration
end
orig_var_num=size(data,2);

% Calculate UIBC
data(:,orig_var_num+1)=data(:,FePos)./data(:,satFePos); %Fe/satFe
data(:,orig_var_num+1) = data(:,orig_var_num+1) - data(:,FePos);

%% Cross-correlation analysis
clr='ygbrb';
sym='....*';
[R, Rpval] = corrplotg(data,grp,clr,sym,'VarNames',variable_name,'testR','on','alpha',0.001);
subplot(4,2,4)
H1=scatter(1,1,1,850,'y.');
hold on
H2=scatter(1,1,1,850,'g.');
H3=scatter(1,1,1,850,'b.');
H4=scatter(1,1,1,100,'b*');
H5=scatter(1,1,1,850,'r.');
hold off
legend([H1 H2 H3 H4 H5],{'healthy controls','without seizures','non-recurrent seizures','non-rec. complex seizures','recurrent seizures'},'location','southwest')
set(gca,'FontSize',14)
axis off
print(fullfile(save_path,'supplfig-crosscorr'), '-dpng', '-r300')
pause(0.2)

%% Univariate analysis: Wilcoxon rank-sum tests and two-sample t-tests (between-group tests)
grp = double(grp);

pW = 5*ones(size(data,2),8);
pT = 5*ones(size(data,2),8);

for vr = 1:size(data,2)
        vec1 = data(grp==1,vr);vec1(isnan(vec1))=[];
        vec2 = data(grp==2,vr);vec2(isnan(vec2))=[];
        vec3 = data(grp==3 | grp == 5,vr);vec3(isnan(vec3))=[];
        vec4 = data(grp==4,vr);vec4(isnan(vec4))=[];
        if ~isempty(vec1) &&  ~isempty(vec2)
                [pW(vr,1), ~] = ranksum(vec1,vec2);
                [~, pT(vr,1)] = ttest2(vec1,vec2);
        else
                pW(vr,1) = NaN;
                pT(vr,1) = NaN;
        end
        if ~isempty(vec1) &&  ~isempty(vec3)
                [pW(vr,2), ~] = ranksum(vec1,vec3);
                [~, pT(vr,2)] = ttest2(vec1,vec3);
        else
                pW(vr,2) = NaN;
                pT(vr,2) = NaN;
        end
        if ~isempty(vec1) &&  ~isempty(vec4)
                [pW(vr,3), ~] = ranksum(vec1,vec4);
                [~, pT(vr,3)] = ttest2(vec1,vec4);
        else
                pW(vr,3) = NaN;
                pT(vr,3) = NaN;
        end
        if ~isempty(vec1) &&  ~isempty(vec4) && ~isempty(vec2) &&  ~isempty(vec3)
                [pW(vr,4), ~] = ranksum([vec1; vec2],[vec3; vec4]);
                [~, pT(vr,4)] = ttest2([vec1; vec2],[vec3; vec4]);
        else
                pW(vr,4) = NaN;
                pT(vr,4) = NaN;
        end
        if ~isempty(vec2) &&  ~isempty(vec3)
                [pW(vr,5), ~] = ranksum(vec2,vec3);
                [~, pT(vr,5)] = ttest2(vec2,vec3);
        else
                pW(vr,5) = NaN;
                pT(vr,5) = NaN;
        end
        if ~isempty(vec2) &&  ~isempty(vec4)
                [pW(vr,6), ~] = ranksum(vec2,vec4);
                [~, pT(vr,6)] = ttest2(vec2,vec4);
        else
                pW(vr,6) = NaN;
                pT(vr,6) = NaN;
        end
        if ~isempty(vec2) &&  ~isempty(vec4) &&  ~isempty(vec3)
                [pW(vr,7), ~] = ranksum(vec2,[vec3; vec4]);
                [~, pT(vr,7)] = ttest2(vec2,[vec3; vec4]);
        else
                pW(vr,7) = NaN;
                pT(vr,7) = NaN;
        end
        if ~isempty(vec3) &&  ~isempty(vec4)
                [pW(vr,8), ~] = ranksum(vec3,vec4);
                [~, pT(vr,8)] = ttest2(vec3,vec4);
        else
                pW(vr,8) = NaN;
                pT(vr,8) = NaN;
        end
end

%% Threshold for Family Wise Error Correction (FWE)
thrfwe=thr_p_uncorr/(size(pW,1)*size(pW,2));

%% Test difference between FS and RFS groups in electrolytes and vitD
fsdata = [mch sodium vitD];
rfsdata = fsdata(grp == 4 & AtOrder > 1,:);
fsdata = fsdata(grp == 4 & AtOrder == 1,:);

fsdata_stat = zeros(size(fsdata,2),3);
rfsdata_stat = zeros(size(fsdata,2),3);
pfsW = 5*ones(size(fsdata,2),1);
for vr = 1:size(fsdata,2)
    [pfsW(vr,1), ~] = ranksum(fsdata(:,vr),rfsdata(:,vr));
    fsdata_stat(vr,:) = quantile(fsdata(:,vr),[0.15 0.5 0.85]);
    rfsdata_stat(vr,:) = quantile(rfsdata(:,vr),[0.15 0.5 0.85]);
end
%% Add sex into analysis
data = [data female];
variable_name{1,end+1} = 'Sex';
%% Add temperature, MCH, vitD and sodium into Model3 modeling
data_model3 = [data temperature mch vitD sodium];
variable_name_model3=variable_name;
variable_name_model3{1,end+1} = 'Temp';
variable_name_model3{1,end+1} = 'MCH';
variable_name_model3{1,end+1} = 'vitD';
variable_name_model3{1,end+1} = 'Sodium';

%% Multivariate analysis: Model3 fitting
% Prepare X and Y matrices and group variable grp_ps_Wilcox
ps=ones(size(grp));
ps(grp<=2)=0; % zero positions where data of control groups are recorded
ps = logical(ps); % identify only positions where FS subjects are recorded
X=data_model3(ps,1:end); % X containing only data of FS subjects
Y=AtNum(ps,1); % Y as idex of attack number
Y_AtOrder = AtOrder(ps,1); % Vector of attack orders
Y(Y>=2) = 2; % Every non-first attack code as group value equal to 2
X_Wilcox = X; % all Y values for between-group testing
Y_Wilcox = Y; % all X values for between-group testing
grp_ps_Wilcox = grp(ps);
nodata = isnan(sum(X,2)); % identify rows where are not collected all data 
Y(nodata==1)=[]; % only Y values where all data are recorded
X(nodata==1,:)=[]; % only X values where all data are recorded

% Possible data nomrlaization (not-used in the manuscript)
% X = ( X - repmat(mean(X),size(X,1),1) ) ./ (  repmat(std(X),size(X,1),1) );
% Y = (Y - mean(Y)) / std(Y);

% ADASYN model3 (Sci Rep revision 1: Q2 of R1)
% ADASYN sythetic dataset matchin female sample size with male sample size
% ADASYN: set up ADASYN parameters an d call the function:

features1f = X(Y==1 & X(:,end-4)==1,[1:end-5 end-3:end]);
features1m = X(Y==1 & X(:,end-4)==0,[1:end-5 end-3:end]);
labels1f = true([size(features1f,1) 1]);
labels1m = false([size(features1m,1) 1]);

features2f = X(Y==2 & X(:,end-4)==1,[1:end-5 end-3:end]);
features2m = X(Y==2 & X(:,end-4)==0,[1:end-5 end-3:end]);
labels2f = true([size(features2f,1) 1]);
labels2m = false([size(features2m,1) 1]);

adasyn_features1                 = [features1f; features1m];
adasyn_labels1                   = [labels1f  ; labels1m  ];
adasyn_beta1                     = [];   %let ADASYN choose default
adasyn_kDensity1                 = [];   %let ADASYN choose default
adasyn_kSMOTE1                   = [];   %let ADASYN choose default
adasyn_featuresAreNormalized1    = false;    %false lets ADASYN handle normalization

adasyn_features2                 = [features2f; features2m];
adasyn_labels2                   = [labels2f  ; labels2m  ];
adasyn_beta2                     = [];   %let ADASYN choose default
adasyn_kDensity2                 = [];   %let ADASYN choose default
adasyn_kSMOTE2                   = [];   %let ADASYN choose default
adasyn_featuresAreNormalized2    = false;    %false lets ADASYN handle normalization


% Divide data regarding groups and sex for dataset undersampling and
% testing regression parameter stability (Sci Rep revision 1: Q2 of R2 and Q1 of R1)
XX0f = X(Y==1 & X(:,end-4)==1,:);
XX0m = X(Y==1 & X(:,end-4)==0,:);
XX1f = X(Y==2 & X(:,end-4)==1,:);
XX1m = X(Y==2 & X(:,end-4)==0,:);
sizeXX0f = size(XX0f,1); sizeXX0f90 = round(0.9*sizeXX0f);  sizeXX0f80 = round(0.8*sizeXX0f);  sizeXX0f70 = round(0.7*sizeXX0f);  sizeXX0f60 = round(0.6*sizeXX0f);  sizeXX0f50 = round(0.5*sizeXX0f);
sizeXX0m = size(XX0m,1); sizeXX0m90 = round(0.9*sizeXX0m);  sizeXX0m80 = round(0.8*sizeXX0m);  sizeXX0m70 = round(0.7*sizeXX0m);  sizeXX0m60 = round(0.6*sizeXX0m);  sizeXX0m50 = round(0.5*sizeXX0m);
sizeXX1f = size(XX1f,1); sizeXX1f90 = round(0.9*sizeXX1f);  sizeXX1f80 = round(0.8*sizeXX1f);  sizeXX1f70 = round(0.7*sizeXX1f);  sizeXX1f60 = round(0.6*sizeXX1f);  sizeXX1f50 = round(0.5*sizeXX1f);
sizeXX1m = size(XX1m,1); sizeXX1m90 = round(0.9*sizeXX1m);  sizeXX1m80 = round(0.8*sizeXX1m);  sizeXX1m70 = round(0.7*sizeXX1m);  sizeXX1m60 = round(0.6*sizeXX1m);  sizeXX1m50 = round(0.5*sizeXX1m);

for iter = 1:niter_adasyn
    % ADASYN (Sci Rep revision 1: Q2 of R1)
    [adasyn_featuresSyn1, adasyn_labelsSyn1] = ADASYN(adasyn_features1, adasyn_labels1, adasyn_beta1, adasyn_kDensity1, adasyn_kSMOTE1, adasyn_featuresAreNormalized1);
    [adasyn_featuresSyn2, adasyn_labelsSyn2] = ADASYN(adasyn_features2, adasyn_labels2, adasyn_beta2, adasyn_kDensity2, adasyn_kSMOTE2, adasyn_featuresAreNormalized2);
    
    XSyn = [X; ...
        [adasyn_featuresSyn1(:,1:end-4), adasyn_labelsSyn1, adasyn_featuresSyn1(:,end-3:end)]; ...
        [adasyn_featuresSyn2(:,1:end-4), adasyn_labelsSyn2, adasyn_featuresSyn2(:,end-3:end)] ];
    YSyn = [Y; ones(size(adasyn_labelsSyn1)); 2*ones(size(adasyn_labelsSyn2))];

    % Step-wise linear regression for model3
    [m3(iter,1).b,m3(iter,1).se,m3(iter,1).pval,m3(iter,1).inmodel,m3(iter,1).stats,m3(iter,1).nextstep,m3(iter,1).history] = stepwisefit(XSyn,YSyn,'penter',0.05);
    Y_predict = sum( repmat(m3(iter,1).b(m3(iter,1).inmodel==1)',size(X,1),1).*X(:,m3(iter,1).inmodel==1) ,2); % Predicted signal Y_predict
    [r, pr]=corrcoef(Y,Y_predict);
    m3(iter,1).Y_predict = Y_predict;
    m3(iter,1).r = r(1,2);
    m3(iter,1).p_r = pr(1,2);
    m3(iter,1).R2 = (1 - m3(iter,1).stats(1,1).SSresid / m3(iter,1).stats(1,1).SStotal)*100;
    tmp=estimate_threshold(Y_predict(Y==2),Y_predict(Y==1),variable_name_model3,999,'he');
    m3(iter,1).threshold = tmp.OptimThr;
    m3(iter,1).sensitivity = tmp.OptimSensitivity;
    m3(iter,1).specificity = tmp.OptimSpecificity;
    
    % Dataset undersampling and testing of regression coefficient stability (Sci Rep revision 1: Q2 of R2 and Q1 of R1)
    if iter == 1
        m390=regress_undersampled_model3(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f90,sizeXX0m90,sizeXX1f90,sizeXX1m90,adasyn_beta1,adasyn_kDensity1,adasyn_kSMOTE1,adasyn_featuresAreNormalized1);
        m380=regress_undersampled_model3(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f80,sizeXX0m80,sizeXX1f80,sizeXX1m80,adasyn_beta1,adasyn_kDensity1,adasyn_kSMOTE1,adasyn_featuresAreNormalized1);
        m370=regress_undersampled_model3(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f70,sizeXX0m70,sizeXX1f70,sizeXX1m70,adasyn_beta1,adasyn_kDensity1,adasyn_kSMOTE1,adasyn_featuresAreNormalized1);
        m360=regress_undersampled_model3(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f60,sizeXX0m60,sizeXX1f60,sizeXX1m60,adasyn_beta1,adasyn_kDensity1,adasyn_kSMOTE1,adasyn_featuresAreNormalized1);
        m350=regress_undersampled_model3(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f50,sizeXX0m50,sizeXX1f50,sizeXX1m50,adasyn_beta1,adasyn_kDensity1,adasyn_kSMOTE1,adasyn_featuresAreNormalized1);
    else
        m390(iter,1)=regress_undersampled_model3(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f90,sizeXX0m90,sizeXX1f90,sizeXX1m90,adasyn_beta1,adasyn_kDensity1,adasyn_kSMOTE1,adasyn_featuresAreNormalized1);
        m380(iter,1)=regress_undersampled_model3(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f80,sizeXX0m80,sizeXX1f80,sizeXX1m80,adasyn_beta1,adasyn_kDensity1,adasyn_kSMOTE1,adasyn_featuresAreNormalized1);
        m370(iter,1)=regress_undersampled_model3(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f70,sizeXX0m70,sizeXX1f70,sizeXX1m70,adasyn_beta1,adasyn_kDensity1,adasyn_kSMOTE1,adasyn_featuresAreNormalized1);
        m360(iter,1)=regress_undersampled_model3(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f60,sizeXX0m60,sizeXX1f60,sizeXX1m60,adasyn_beta1,adasyn_kDensity1,adasyn_kSMOTE1,adasyn_featuresAreNormalized1);
        m350(iter,1)=regress_undersampled_model3(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f50,sizeXX0m50,sizeXX1f50,sizeXX1m50,adasyn_beta1,adasyn_kDensity1,adasyn_kSMOTE1,adasyn_featuresAreNormalized1);
    end
end

M3(1,1).inmodel = reshape([m3(:,1).inmodel]',size(m3(1,1).inmodel,2),niter_adasyn)';
M3(1,1).inmodel_unique = unique(M3.inmodel,'rows');
M3(1,1).inmodel_unique_prob = zeros(size(M3.inmodel_unique,1),1);

for res = 1:size(M3.inmodel_unique,1)
    for iter = 1:niter_adasyn
        M3.inmodel_unique_prob(res,1) = M3.inmodel_unique_prob(res,1) + ...
            double(sum(M3.inmodel_unique(res,:) == M3.inmodel(iter,:)) == size(m3(1,1).inmodel,2));
    end
end
M3(1,1).inmodel_unique_prob = 100*M3.inmodel_unique_prob/niter_adasyn;
M3(1,1).best = find(M3.inmodel_unique_prob==max(M3.inmodel_unique_prob));
M3(1,1).pos = sum(M3.inmodel == M3.inmodel_unique(M3.best,:),2) == size(M3.inmodel,2);
M3(1,1).varid = find(M3.inmodel_unique(M3.best,:)==1);
b = [m3.b]';
b = b(M3.pos,M3.varid);
M3.b(:,1) =  mean(b);
M3.b(:,2) =  std(b);
pval = [m3.pval]';
pval = pval(M3.pos,M3.varid);
M3.pval = median(pval);
r = [m3(M3.pos,1).r]';
M3.r = [mean(r) std(r)];
M3.p_r = median([m3(M3.pos,1).p_r]);
R2 = [m3(M3.pos,1).R2]';
M3.R2 = [mean(R2) std(R2)];
M3.prob = M3.inmodel_unique_prob(M3.best);
M3.unique_models = size(M3.inmodel_unique,1);
th = [m3(M3.pos,1).threshold]';
M3.threshold = [mean(th) std(th)];
th = [m3(M3.pos,1).sensitivity]';
M3.sensitivity = [mean(th) std(th)];
th = [m3(M3.pos,1).specificity]';
M3.specificity = [mean(th) std(th)];
th = arrayfun(@(x) x.stats.rmse,m3); th = th(M3.pos,1);
M3.rmse = [mean(th) std(th)];
th = arrayfun(@(x) x.stats.fstat,m3); th = th(M3.pos,1);
M3.fstat = [mean(th) std(th)];

M390 = get_downsamp_model_info(m390,M3,niter_adasyn);
M380 = get_downsamp_model_info(m380,M3,niter_adasyn);
M370 = get_downsamp_model_info(m370,M3,niter_adasyn);
M360 = get_downsamp_model_info(m360,M3,niter_adasyn);
M350 = get_downsamp_model_info(m350,M3,niter_adasyn);

% Predicted signal Y_predict
% Y_predict = sum( repmat(M3.b(:,1)',size(X,1),1).*X(:,M3.varid) ,2); % Predicted signal for record where all data were acquired
% [model3_r, model3_pr]=corrcoef(Y,Y_predict);
% model3_R2 = (1 - model3_stats(1,1).SSresid / model3_stats(1,1).SStotal)*100;
 
var_sig=M3.varid;

Yp=sum(X_Wilcox(:,M3.varid).*repmat(M3.b(:,1)',size(X_Wilcox,1),1),2); % Predicted signal for all FS subjects
p_Yp_W = ranksum(Yp(Y_Wilcox==1),Yp(Y_Wilcox==2)); % Wilcoxon rank-sum test
[~, p_Yp_T] = ttest2(Yp(Y_Wilcox==1),Yp(Y_Wilcox==2)); % Two-sample t-test

% Sub-grouping of the predicted signal for visualization purposes 
Yp1=Yp(Y_Wilcox==1);
Yp2=Yp(Y_Wilcox==2);
Yp1_mean = median(Yp1,'omitnan');
Yp2_mean = median(Yp2,'omitnan');
Yp1_1=Yp(grp_ps_Wilcox==3);
Yp1_2=Yp(grp_ps_Wilcox==5);
Yp2_1=Yp(Y_Wilcox==2 & Y_AtOrder==1);
Yp2_2=Yp(Y_Wilcox==2 & Y_AtOrder>1);

Yp1_Cinterval = quantile(Yp1,[0.025 0.25 0.50 0.75 0.975]);
Yp2_Cinterval = quantile(Yp2,[0.025 0.25 0.50 0.75 0.975]);

% Model3 sensitivity and specificity analysis
sts=estimate_threshold(Yp2,Yp1,variable_name_model3,999,'he');idx=idx+1;
S_risk(1,:)=[sts(1,idx-1).OptimThr sts(1,idx-1).OptimSensitivity sts(1,idx-1).OptimSpecificity];
seiz_risk=[sts(1,idx-1).OptimThr sts(1,idx-1).OptimSensitivity sts(1,idx-1).OptimSpecificity];

% Model3 visualization
h(2).fig = figure(2);
set(h(2).fig,'Position',[50 50 1300 500])
subplot(1,2,1)
scatter3(data_model3(grp==3,var_sig(1)),data_model3(grp==3,var_sig(2)),data_model3(grp==3,var_sig(3)),850, 'b.')
hold on
scatter3(data_model3(grp==5,var_sig(1)),data_model3(grp==5,var_sig(2)),data_model3(grp==5,var_sig(3)),100, 'b*','LineWidth',2)
scatter3(data_model3(grp==4 & AtOrder==1,var_sig(1)),data_model3(grp==4 & AtOrder==1,var_sig(2)),data_model3(grp==4 & AtOrder==1,var_sig(3)),850, 'r.')
scatter3(data_model3(grp==4 & AtOrder>1,var_sig(1)),data_model3(grp==4 & AtOrder>1,var_sig(2)),data_model3(grp==4 & AtOrder>1,var_sig(3)),100, 'r*','LineWidth',2)
% xxdata = [data_model3(grp==3,var_sig(1)); data_model3(grp==4,var_sig(1)); data_model3(grp==5,var_sig(1))];
% yydata = [data_model3(grp==3,var_sig(2)); data_model3(grp==4,var_sig(2)); data_model3(grp==5,var_sig(2))];
% par = polyfit(xxdata,yydata,1);
Xlimits = get(gca,'XLim');
Ylimits = get(gca,'YLim');
Zlimits = get(gca,'ZLim');
% Yvals = par(1).*Xlimits + par(2);
% rr = corrcoef(xxdata,yydata);
% plot(Xlimits,Yvals,'-.','LineWidth',2,'Color',[80 80 80]/255)
xlim(Xlimits)
ylim(Ylimits)
zlim(Zlimits)
% text(35,118,['r=' num2str(rr(1,2),'%10.3f')],'Fontsize',14,'HorizontalAlignment','left')
hold off
grid on
xlabel([variable_name_model3{1,var_sig(1)} ' [%]'])
ylabel([variable_name_model3{1,var_sig(2)} ' [g/l]'])
zlabel(variable_name_model3{1,var_sig(3)})
legend('non-recurrent seizures','non-rec. complex seizures','recurrent seizures (1^{st} seizure)','recurrent seizures (repeated seizure)','Location','southeast')
set(gca,'FontSize',14,...
        'LineWidth',2)

subplot(1,2,2)
H3=plot([0 5],[seiz_risk(1,1) seiz_risk(1,1)],':','LineWidth',4,'Color',[0.5 0.5 0.5]);
hold on
plot([0.27 0.73],[Yp1_Cinterval(2) Yp1_Cinterval(2)],'k','LineWidth',3);
plot([0.27 0.73],[Yp1_Cinterval(4) Yp1_Cinterval(4)],'k','LineWidth',3);
plot([0.27 0.27],[Yp1_Cinterval(2) Yp1_Cinterval(4)],'k','LineWidth',3);
plot([0.73 0.73],[Yp1_Cinterval(2) Yp1_Cinterval(4)],'k','LineWidth',3);
plot([0.5 0.5],[Yp1_Cinterval(1) Yp1_Cinterval(2)],'k','LineWidth',3);
plot([0.48 0.52],[Yp1_Cinterval(1) Yp1_Cinterval(1)],'k','LineWidth',3);
plot([0.5 0.5],[Yp1_Cinterval(4) Yp1_Cinterval(5)],'k','LineWidth',3);
plot([0.48 0.52],[Yp1_Cinterval(5) Yp1_Cinterval(5)],'k','LineWidth',3);
plot([0.77 1.23],[Yp2_Cinterval(2) Yp2_Cinterval(2)],'k','LineWidth',3);
plot([0.77 1.23],[Yp2_Cinterval(4) Yp2_Cinterval(4)],'k','LineWidth',3);
plot([0.77 0.77],[Yp2_Cinterval(2) Yp2_Cinterval(4)],'k','LineWidth',3);
plot([1.23 1.23],[Yp2_Cinterval(2) Yp2_Cinterval(4)],'k','LineWidth',3);
plot([1.0 1.0],[Yp2_Cinterval(1) Yp2_Cinterval(2)],'k','LineWidth',3);
plot([0.98 1.02],[Yp2_Cinterval(1) Yp2_Cinterval(1)],'k','LineWidth',3);
plot([1.0 1.0],[Yp2_Cinterval(4) Yp2_Cinterval(5)],'k','LineWidth',3);
plot([0.98 1.02],[Yp2_Cinterval(5) Yp2_Cinterval(5)],'k','LineWidth',3);
H1 = plot([0.27 0.73],[Yp1_mean Yp1_mean],'k-.','LineWidth',3);
H2 = plot([0.77 1.23],[Yp2_mean Yp2_mean],'k-.','LineWidth',3);
scatter(1*ones(size(Yp1_1))/2,Yp1_1,850, 'b.', 'jitter','on', 'jitterAmount', 0.14,'MarkerEdgeAlpha',0.7,'MarkerFaceAlpha',0.7);
scatter(1*ones(size(Yp1_2))/2,Yp1_2,100, 'b*', 'jitter','on', 'jitterAmount', 0.14,'MarkerEdgeAlpha',0.7,'MarkerFaceAlpha',0.7,'LineWidth',2);
scatter(2*ones(size(Yp2_1))/2,Yp2_1,850, '.','MarkerEdgeColor',[1 0 0], 'jitter','on', 'jitterAmount', 0.14,'MarkerEdgeAlpha',0.7,'MarkerFaceAlpha',0.7);
scatter(2*ones(size(Yp2_2))/2,Yp2_2,100, 'r*', 'jitter','on', 'jitterAmount', 0.14,'MarkerEdgeAlpha',0.7,'MarkerFaceAlpha',0.7,'LineWidth',2);
plot([0.5 1],[2.14 2.14],'k-.','LineWidth',2)
plot([0.5 0.5],[2.08 2.14],'k-.','LineWidth',2)
plot([1 1],[2.08 2.14],'k-.','LineWidth',2)
plot([0.75 0.75],[2.14 2.20],'k-.','LineWidth',2)
text(0.75,2.23,'*','FontSize',20,'HorizontalAlignment','center')
text(0.77,2.2,['p=' num2str(p_Yp_W,'%10.5f')],'FontSize',14,'HorizontalAlignment','left')
hold off
grid on
ylabel({[num2str(M3.b(1,1),'%10.4f') '*' variable_name_model3{1,var_sig(1)} '+' num2str(M3.b(2,1),'%10.4f') '*' variable_name_model3{1,var_sig(2)}]; ['+' num2str(M3.b(3,1),'%10.4f') '*' variable_name_model3{1,var_sig(3)} ]} )
ylim([0.6 2.3])
xlim([0.2 1.3])
legend([H1, H3],{'median value',['thr=' num2str(seiz_risk(1,1),'%10.2f') ';SE=' num2str(seiz_risk(1,2),'%10.1f') '%;SP=' num2str(seiz_risk(1,3),'%10.1f') '%']},'location','southeast')
set(gca,'XTick',(1:1:2)/2,...
         'XTickLabel',{'non-recurrent seizures'
                       'recurrent seizures'
                       },...
         'TickLength',[0 0],'LineWidth',2,...
         'FontSize',14)
xtickangle(14)
print(fullfile(save_path,'fig2c'), '-dpng', '-r300')
pause(0.2)

%% Multivariate analysis: Model1 fitting
ps=~isnan(sum(data,2)); % Positions where all data are recorded
X=data(ps,1:end); % Matrix X of all recorded data
grp_ps= grp(ps); % Group categories on positions where all data are recorded
Y=zeros(size(X,1),1); % Zeros for all positions of healthy controls
Y(grp_ps==2) = 1; % Value 1 at all positions of all febrile patients without seizures
Y(grp_ps==3) = 2; % Value 2 for all positions of all febrile patients with seizures
Y(grp_ps==4) = 2;
Y(grp_ps==5) = 2;
X(:,satFePos) = randn(size(X,1),1); % Substitute satFe data with Gausian random variable to exclude satFe from the fitting.
% If satFe is preserved in analysis model1 output is very similar but with
% worse specificity. Exlusion satFe from analysis improves model1 specificity.
% You can comment the previous command line and make the experiment yourself.

% ADASYN model1 (Sci Rep revision 1: Q2 of R1)
% ADASYN sythetic dataset matchin female sample size with male sample size
% ADASYN: set up ADASYN parameters and call the function:

featuresf = X(Y==2 & X(:,end)==1,1:end-1);
featuresm = X(Y==2 & X(:,end)==0,1:end-1);
labelsf = true([size(featuresf,1) 1]);
labelsm = false([size(featuresm,1) 1]);

adasyn_features                 = [featuresf; featuresm];
adasyn_labels                   = [labelsf  ; labelsm  ];
adasyn_beta                     = [];   %let ADASYN choose default
adasyn_kDensity                 = [];   %let ADASYN choose default
adasyn_kSMOTE                   = [];   %let ADASYN choose default
adasyn_featuresAreNormalized    = false;    %false lets ADASYN handle normalization

% Divide data regarding groups and sex for dataset undersampling and
% testing regression parameter stability (Sci Rep revision 1: Q2 of R2 and Q1 of R1)
XX0f = X(Y==0 & X(:,end)==1,:);
XX0m = X(Y==0 & X(:,end)==0,:);
XX1f = X(Y==1 & X(:,end)==1,:);
XX1m = X(Y==1 & X(:,end)==0,:);
XX2f = X(Y==2 & X(:,end)==1,:);
XX2m = X(Y==2 & X(:,end)==0,:);
sizeXX0f = size(XX0f,1); sizeXX0f90 = round(0.9*sizeXX0f);  sizeXX0f80 = round(0.8*sizeXX0f);  sizeXX0f70 = round(0.7*sizeXX0f);  sizeXX0f60 = round(0.6*sizeXX0f);  sizeXX0f50 = round(0.5*sizeXX0f);
sizeXX0m = size(XX0m,1); sizeXX0m90 = round(0.9*sizeXX0m);  sizeXX0m80 = round(0.8*sizeXX0m);  sizeXX0m70 = round(0.7*sizeXX0m);  sizeXX0m60 = round(0.6*sizeXX0m);  sizeXX0m50 = round(0.5*sizeXX0m);
sizeXX1f = size(XX1f,1); sizeXX1f90 = round(0.9*sizeXX1f);  sizeXX1f80 = round(0.8*sizeXX1f);  sizeXX1f70 = round(0.7*sizeXX1f);  sizeXX1f60 = round(0.6*sizeXX1f);  sizeXX1f50 = round(0.5*sizeXX1f);
sizeXX1m = size(XX1m,1); sizeXX1m90 = round(0.9*sizeXX1m);  sizeXX1m80 = round(0.8*sizeXX1m);  sizeXX1m70 = round(0.7*sizeXX1m);  sizeXX1m60 = round(0.6*sizeXX1m);  sizeXX1m50 = round(0.5*sizeXX1m);
sizeXX2f = size(XX2f,1); sizeXX2f90 = round(0.9*sizeXX2f);  sizeXX2f80 = round(0.8*sizeXX2f);  sizeXX2f70 = round(0.7*sizeXX2f);  sizeXX2f60 = round(0.6*sizeXX2f);  sizeXX2f50 = round(0.5*sizeXX2f);
sizeXX2m = size(XX2m,1); sizeXX2m90 = round(0.9*sizeXX2m);  sizeXX2m80 = round(0.8*sizeXX2m);  sizeXX2m70 = round(0.7*sizeXX2m);  sizeXX2m60 = round(0.6*sizeXX2m);  sizeXX2m50 = round(0.5*sizeXX2m);

for iter = 1:niter_adasyn
    % ADASYN (Sci Rep revision 1: Q2 of R1)
    [adasyn_featuresSyn, adasyn_labelsSyn] = ADASYN(adasyn_features, adasyn_labels, adasyn_beta, adasyn_kDensity, adasyn_kSMOTE, adasyn_featuresAreNormalized);
    
    XSyn = [X; [adasyn_featuresSyn, adasyn_labelsSyn]];
    YSyn = [Y; 2*ones(size(adasyn_labelsSyn))];

    % Step-wise linear regression for model1
    [m1(iter,1).b,m1(iter,1).se,m1(iter,1).pval,m1(iter,1).inmodel,m1(iter,1).stats,m1(iter,1).nextstep,m1(iter,1).history] = stepwisefit(XSyn,YSyn,'penter',0.05);
    Y_predict = sum( repmat(m1(iter,1).b(m1(iter,1).inmodel==1)',size(X,1),1).*X(:,m1(iter,1).inmodel==1) ,2); % Predicted signal Y_predict
    [r, pr]=corrcoef(Y,Y_predict);
    m1(iter,1).Y_predict = Y_predict;
    m1(iter,1).r = r(1,2);
    m1(iter,1).p_r = pr(1,2);
    m1(iter,1).R2 = (1 - m1(iter,1).stats(1,1).SSresid / m1(iter,1).stats(1,1).SStotal)*100;
    tmp=estimate_threshold(Y_predict(Y==2),Y_predict(Y==1),variable_name,997,'he');
    m1(iter,1).threshold = tmp.OptimThr;
    m1(iter,1).sensitivity = tmp.OptimSensitivity;
    m1(iter,1).specificity = tmp.OptimSpecificity;
    
    % Dataset undersampling and testing of regression coefficient stability (Sci Rep revision 1: Q2 of R2 and Q1 of R1)
    if iter == 1
        m190=regress_undersampled_model1(XX0f,XX0m,XX1f,XX1m,XX2f,XX2m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX2f,sizeXX2m,sizeXX0f90,sizeXX0m90,sizeXX1f90,sizeXX1m90,sizeXX2f90,sizeXX2m90,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
        m180=regress_undersampled_model1(XX0f,XX0m,XX1f,XX1m,XX2f,XX2m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX2f,sizeXX2m,sizeXX0f80,sizeXX0m80,sizeXX1f80,sizeXX1m80,sizeXX2f80,sizeXX2m80,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
        m170=regress_undersampled_model1(XX0f,XX0m,XX1f,XX1m,XX2f,XX2m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX2f,sizeXX2m,sizeXX0f70,sizeXX0m70,sizeXX1f70,sizeXX1m70,sizeXX2f70,sizeXX2m70,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
        m160=regress_undersampled_model1(XX0f,XX0m,XX1f,XX1m,XX2f,XX2m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX2f,sizeXX2m,sizeXX0f60,sizeXX0m60,sizeXX1f60,sizeXX1m60,sizeXX2f60,sizeXX2m60,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
        m150=regress_undersampled_model1(XX0f,XX0m,XX1f,XX1m,XX2f,XX2m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX2f,sizeXX2m,sizeXX0f50,sizeXX0m50,sizeXX1f50,sizeXX1m50,sizeXX2f50,sizeXX2m50,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
    else
        m190(iter,1)=regress_undersampled_model1(XX0f,XX0m,XX1f,XX1m,XX2f,XX2m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX2f,sizeXX2m,sizeXX0f90,sizeXX0m90,sizeXX1f90,sizeXX1m90,sizeXX2f90,sizeXX2m90,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
        m180(iter,1)=regress_undersampled_model1(XX0f,XX0m,XX1f,XX1m,XX2f,XX2m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX2f,sizeXX2m,sizeXX0f80,sizeXX0m80,sizeXX1f80,sizeXX1m80,sizeXX2f80,sizeXX2m80,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
        m170(iter,1)=regress_undersampled_model1(XX0f,XX0m,XX1f,XX1m,XX2f,XX2m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX2f,sizeXX2m,sizeXX0f70,sizeXX0m70,sizeXX1f70,sizeXX1m70,sizeXX2f70,sizeXX2m70,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
        m160(iter,1)=regress_undersampled_model1(XX0f,XX0m,XX1f,XX1m,XX2f,XX2m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX2f,sizeXX2m,sizeXX0f60,sizeXX0m60,sizeXX1f60,sizeXX1m60,sizeXX2f60,sizeXX2m60,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
        m150(iter,1)=regress_undersampled_model1(XX0f,XX0m,XX1f,XX1m,XX2f,XX2m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX2f,sizeXX2m,sizeXX0f50,sizeXX0m50,sizeXX1f50,sizeXX1m50,sizeXX2f50,sizeXX2m50,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
    end
end

M1(1,1).inmodel = reshape([m1(:,1).inmodel]',size(m1(1,1).inmodel,2),niter_adasyn)';
M1(1,1).inmodel_unique = unique(M1.inmodel,'rows');
M1(1,1).inmodel_unique_prob = zeros(size(M1.inmodel_unique,1),1);

for res = 1:size(M1.inmodel_unique,1)
    for iter = 1:niter_adasyn
        M1.inmodel_unique_prob(res,1) = M1.inmodel_unique_prob(res,1) + ...
            double(sum(M1.inmodel_unique(res,:) == M1.inmodel(iter,:)) == size(m1(1,1).inmodel,2));
    end
end
M1(1,1).inmodel_unique_prob = 100*M1.inmodel_unique_prob/niter_adasyn;
M1(1,1).best = find(M1.inmodel_unique_prob==max(M1.inmodel_unique_prob));
M1(1,1).pos = sum(M1.inmodel == M1.inmodel_unique(M1.best,:),2) == size(M1.inmodel,2);
M1(1,1).varid = find(M1.inmodel_unique(M1.best,:)==1);
b = [m1.b]';
b = b(M1.pos,M1.varid);
M1.b(:,1) =  mean(b);
M1.b(:,2) =  std(b);
pval = [m1.pval]';
pval = pval(M1.pos,M1.varid);
M1.pval = median(pval);
r = [m1(M1.pos,1).r]';
M1.r = [mean(r) std(r)];
M1.p_r = median([m1(M1.pos,1).p_r]);
R2 = [m1(M1.pos,1).R2]';
M1.R2 = [mean(R2) std(R2)];
M1.prob = M1.inmodel_unique_prob(M1.best);
M1.unique_models = size(M1.inmodel_unique,1);
th = [m1(M1.pos,1).threshold]';
M1.threshold = [mean(th) std(th)];
th = [m1(M1.pos,1).sensitivity]';
M1.sensitivity = [mean(th) std(th)];
th = [m1(M1.pos,1).specificity]';
M1.specificity = [mean(th) std(th)];
th = arrayfun(@(x) x.stats.rmse,m1); th = th(M1.pos,1);
M1.rmse = [mean(th) std(th)];
th = arrayfun(@(x) x.stats.fstat,m1); th = th(M1.pos,1);
M1.fstat = [mean(th) std(th)];

M190 = get_downsamp_model_info(m190,M1,niter_adasyn);
M180 = get_downsamp_model_info(m180,M1,niter_adasyn);
M170 = get_downsamp_model_info(m170,M1,niter_adasyn);
M160 = get_downsamp_model_info(m160,M1,niter_adasyn);
M150 = get_downsamp_model_info(m150,M1,niter_adasyn);

% Step-wise linear regression for model1
% [model1_b,model1_se,model1_pval,model1_inmodel,model1_stats,model1_nextstep,model1_history] =stepwisefit(X,Y,'penter',0.05);
Y_predict = sum( repmat(M1.b(:,1)',size(X,1),1).*X(:,M1.varid) ,2); % Predicted signal Y_predict
Predicted = Y_predict; % Store predicted signal for further analysis
% [model1_r, model1_pr]=corrcoef(Y,Y_predict);
% model1_R2 = (1 - model1_stats(1,1).SSresid / model1_stats(1,1).SStotal)*100;

% Sorted variables based on variable significance to the model
var_sig=M1.varid;
[~,I] = sort(M1.pval);
var_sig=var_sig(I);
model1_b = M1.b(I);

% Sub-grouping of the predicted signal for betweem-group testings and for visualization purposes 
Yp1=Y_predict(grp_ps==1);
Yp2=Y_predict(grp_ps==2);
Yp3=Y_predict(grp_ps==3 | grp_ps == 5);
Yp4=Y_predict(grp_ps==4);
Yp3_1=Y_predict(grp_ps==3);
Yp3_2=Y_predict(grp_ps==5);

% Wilcoxon rank-sum tests
p_Yp14_W(1,1) = ranksum(Yp1,Yp2);
p_Yp14_W(1,2) = ranksum(Yp1,Yp3);
p_Yp14_W(1,3) = ranksum(Yp1,Yp4);
p_Yp14_W(1,4) = ranksum([Yp1; Yp2],[Yp3; Yp4]);
p_Yp14_W(1,5) = ranksum(Yp2,Yp3);
p_Yp14_W(1,6) = ranksum(Yp2,Yp4);
p_Yp14_W(1,7) = ranksum(Yp2,[Yp3; Yp4]);
p_Yp14_W(1,8) = ranksum(Yp3,Yp4);

% Two-sample t-tests
[~, p_Yp14_T(1,1)] = ttest2(Yp1,Yp2);
[~, p_Yp14_T(1,2)] = ttest2(Yp1,Yp3);
[~, p_Yp14_T(1,3)] = ttest2(Yp1,Yp4);
[~, p_Yp14_T(1,4)] = ttest2([Yp1; Yp2],[Yp3; Yp4]);
[~, p_Yp14_T(1,5)] = ttest2(Yp2,Yp3);
[~, p_Yp14_T(1,6)] = ttest2(Yp2,Yp4);
[~, p_Yp14_T(1,7)] = ttest2(Yp2,[Yp3; Yp4]);
[~, p_Yp14_T(1,8)] = ttest2(Yp3,Yp4);

% Sensitivity and specificity analysis
seiz = [Yp3; Yp4];
nonseiz=Yp2;
sts(1,idx)=estimate_threshold(seiz,nonseiz,variable_name,997,'he');idx=idx+1;
S_risk(1,:)=[sts(1,idx-1).OptimThr sts(1,idx-1).OptimSensitivity sts(1,idx-1).OptimSpecificity];

% Model1 visualization
h(3).fig = figure(3);
set(h(3).fig,'Position',[50 50 1300 500])
subplot(5,2,[1 3 5 7])
if size(var_sig,2)<3
    scatter(data(grp==1,var_sig(1)),data(grp==1,var_sig(2)),850, 'y.')
    hold on
    scatter(data(grp==2,var_sig(1)),data(grp==2,var_sig(2)),850, 'g.')
    scatter(data(grp==3,var_sig(1)),data(grp==3,var_sig(2)),850, 'b.')
    scatter(data(grp==5,var_sig(1)),data(grp==5,var_sig(2)),100, 'b*','LineWidth',2)
    scatter(data(grp==4,var_sig(1)),data(grp==4,var_sig(2)),850, 'r.')
else
    scatter3(data(grp==1,var_sig(1)),data(grp==1,var_sig(2)),data(grp==1,var_sig(3)),850, 'y.')
    hold on
    scatter3(data(grp==2,var_sig(1)),data(grp==2,var_sig(2)),data(grp==2,var_sig(3)),850, 'g.')
    scatter3(data(grp==3,var_sig(1)),data(grp==3,var_sig(2)),data(grp==3,var_sig(3)),850, 'b.')
    scatter3(data(grp==5,var_sig(1)),data(grp==5,var_sig(2)),data(grp==5,var_sig(3)),100, 'b*','LineWidth',2)
    scatter3(data(grp==4,var_sig(1)),data(grp==4,var_sig(2)),data(grp==4,var_sig(3)),850, 'r.')
end
hold off
grid on
xlabel([variable_name{1,var_sig(1)} ' [\mumol/l]'])
ylabel([variable_name{1,var_sig(2)} ' [\mumol/l]'])
if size(var_sig,2)>=3
    zlabel([variable_name{1,var_sig(3)} ' [%]'])
end
set(gca,'FontSize',14,...
        'LineWidth',2)

subplot(1,2,2)
if size(var_sig,2)<3
    YLBL1=[num2str(model1_b(1),'%10.3f') '*' variable_name{1,var_sig(1)}  ...
             '+' num2str(model1_b(2),'%10.3f') '*' variable_name{1,var_sig(2)}];
elseif size(var_sig,2)==3
    YLBL1=[num2str(model1_b(1),'%10.3f') '*' variable_name{1,var_sig(1)}  ...
            '+' num2str(model1_b(2),'%10.3f') '*' variable_name{1,var_sig(2)}...
            num2str(model1_b(3),'%10.3f') '*' variable_name{1,var_sig(3)} ];
else
    YLBL1={[num2str(model1_b(1),'%10.3f') '*' variable_name{1,var_sig(1)}  ...
            '+' num2str(model1_b(2),'%10.3f') '*' variable_name{1,var_sig(2)}],...
            ['+' num2str(model1_b(3),'%10.3f') '*' variable_name{1,var_sig(3)} ...
            '+' num2str(model1_b(4),'%10.3f') '*' variable_name{1,var_sig(4)} ]};
end
plot_cat_scatter(Yp1,Yp2,Yp3_1,Yp3_2,Yp4,YLBL1,'southeast',S_risk(1,:))

subplot(6,2,11)
plot([-2 -1],[0 1])
xlim([0 1])
text(0.4725,0.60,'Sub-group difference testings with Wilcoxon rank-sum test','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','center')
text(0.00,0.25,'1vs2','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.15,0.25,'1vs3','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.30,0.25,'1vs4','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.45,0.25,'12vs34','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.60,0.25,'2vs3','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.75,0.25,'2vs4','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.90,0.25,'2vs34','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(1.05,0.25,'3vs4','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
xp = 0.00:0.15:1.05;
for xi = 1:size(p_Yp14_W,2)
        p = p_Yp14_W(1,xi);
        clr = color_decision(p,thrfwe);
        if p<0.00005
                prec='%10.0e';
        else
                prec='%10.4f';
        end
        if p<thrfwe
                fntwght='bold';
        else
                fntwght='normal';
        end
        text(xp(xi),-0.10,num2str(p,prec),'FontSize',fntSiz,'FontWeight',fntwght,'HorizontalAlignment','right','Color',clr)
end
text(-0.13,-0.45,'Green-bold-highlighted p-values fulfill the corrected significance','FontSize',fntSiz,'FontWeight','normal','HorizontalAlignment','left')
text(-0.13,-0.80,'criterion p_{FWE}<0.05 (Family Wise Error correction).','FontSize',fntSiz,'FontWeight','normal','HorizontalAlignment','left')
axis off

print(fullfile(save_path,'fig2a'), '-dpng', '-r300')
pause(0.2)

%% Univariate analysis: visualization and sesitivity/specificity analysis
h(4).fig = figure(4);
set(h(4).fig,'Position',[50 10 1250 1400])
subplot(3,2,1)
varID=satFePos;
seiz=[data(grp==3,varID);data(grp==5,varID);data(grp==4,varID)];
nonseiz=data(grp==2,varID);
sts(1,idx)=estimate_threshold(seiz,nonseiz,variable_name,varID,'le');idx=idx+1;
seiz_risk=[sts(1,idx-1).OptimThr sts(1,idx-1).OptimSensitivity sts(1,idx-1).OptimSpecificity];
plot_cat_scatter(data(grp==1,varID),data(grp==2,varID),data(grp==3,varID),data(grp==5,varID),data(grp==4,varID),variable_name{1,varID},'northeast',seiz_risk)
% [Usatfe, ~] = randomize_univariate_separation(data(:,varID),grp,variable_name,varID,'le',female,niter_adasyn);

subplot(3,2,3)
varID=FePos;
seiz=[data(grp==3,varID);data(grp==5,varID);data(grp==4,varID)];
nonseiz=data(grp==2,varID);
sts(1,idx)=estimate_threshold(seiz,nonseiz,variable_name,varID,'le');idx=idx+1;
seiz_risk=[sts(1,idx-1).OptimThr sts(1,idx-1).OptimSensitivity sts(1,idx-1).OptimSpecificity];
plot_cat_scatter(data(grp==1,varID),data(grp==2,varID),data(grp==3,varID),data(grp==5,varID),data(grp==4,varID),[variable_name{1,varID} ' [\mumol/l]'],' ',seiz_risk)
% [Ufe, ~] = randomize_univariate_separation(data(:,varID),grp,variable_name,varID,'le',female,niter_adasyn);

subplot(3,2,4)
varID=find(strcmp(variable_name,'UIBC')==1);
seiz=[data(grp==3,varID);data(grp==5,varID);data(grp==4,varID)];
nonseiz=data(grp==2,varID);
sts(1,idx)=estimate_threshold(seiz,nonseiz,variable_name,varID,'he');idx=idx+1;
seiz_risk=[sts(1,idx-1).OptimThr sts(1,idx-1).OptimSensitivity sts(1,idx-1).OptimSpecificity];
plot_cat_scatter(data(grp==1,varID),data(grp==2,varID),data(grp==3,varID),data(grp==5,varID),data(grp==4,varID),[variable_name{1,varID} ' [\mumol/l]' ],' ',seiz_risk)
% [Uuibc, ~] = randomize_univariate_separation(data(:,varID),grp,variable_name,varID,'he',female,niter_adasyn);

subplot(3,2,5)
varID=find(strcmp(variable_name,'Height')==1);
seiz_risk=[];
plot_cat_scatter(data(grp==1,varID),data(grp==2,varID),data(grp==3,varID),data(grp==5,varID),data(grp==4,varID),[variable_name{1,varID} ' [%]' ],' ',seiz_risk)

subplot(3,2,6)
varID=find(strcmp(variable_name,'HGB')==1);
seiz_risk=[];
plot_cat_scatter(data(grp==1,varID),data(grp==2,varID),data(grp==3,varID),data(grp==5,varID),data(grp==4,varID),[variable_name{1,varID} ' [g/l]' ],' ',seiz_risk)

subplot(3,2,2)
plot([-2 -1],[0 1])
xlim([0 1])
text(0.5,0.97,'Sub-group difference testings with Wilcoxon rank-sum test','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','center')
text(-0.05,0.90,'Groups','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.10,0.90,'1vs2','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.25,0.90,'1vs3','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.40,0.90,'1vs4','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.55,0.90,'12vs34','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.70,0.90,'2vs3','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.85,0.90,'2vs4','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(1.00,0.90,'2vs34','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(1.15,0.90,'3vs4','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')

yp = 0.83:-0.07:-0.15;
xp = -0.05:0.15:1.15;
for yi = 1:size(pW,1)
        for xi = 1:size(pW,2)+1
                if xp(xi)==-0.05
                         text(-0.05,yp(yi),variable_name{1,yi},'FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
                else
                        p = pW(yi,xi-1);
                        clr = color_decision(p,thrfwe);
                        if p<0.00005
                                prec='%10.0e';
                        else
                                prec='%10.4f';
                        end
                        if p<thrfwe
                                fntwght='bold';
                        else
                                fntwght='normal';
                        end
                        text(xp(xi),yp(yi),num2str(p,prec),'FontSize',fntSiz,'FontWeight',fntwght,'HorizontalAlignment','right','Color',clr)
                end
        end
end
text(-0.15,0.27-0.14,'Green-bold-highlighted p-values fulfill the corrected significance','FontSize',fntSiz,'FontWeight','normal','HorizontalAlignment','left')
text(-0.15,0.20-0.14,'criterion p_{FWE}<0.05 (Family Wise Error correction).','FontSize',fntSiz,'FontWeight','normal','HorizontalAlignment','left')
text(-0.15,0.13-0.14,'Black-highlighted p-values fulfill the significance criterion p<0.05','FontSize',fntSiz,'FontWeight','normal','HorizontalAlignment','left')
axis off

print(fullfile(save_path,'fig1'), '-dpng', '-r300')
pause(0.2)
%% Multivariate analysis: Model2 fitting
ps=~isnan(sum(data,2)); % Positions where all data are recorded
X=data(ps,1:end); % Matrix X of all recorded data
X2=X; % Matrix X2 of all recorded data
grp_ps= grp(ps); % Group categories on positions where all data are recorded
Y=zeros(size(X,1),1); % Zeros for all positions of healthy controls
Y(grp_ps==2) = 1; % Value 1 at all positions of all febrile patients without seizures
Y(grp_ps==3) = 2; % Value 2 for all positions of all febrile patients with seizures
Y(grp_ps==4) = 2;
Y(grp_ps==5) = 2;
grp_ps2 = grp_ps;
grp_ps2(grp_ps==1)=[]; % Group categories on positions where all data are recorded without afebrille healthy controls
X(Y==0,:)=[]; % Matrix X of all recorded data without data of afebrille healthy controls
Y(Y==0)=[]; Y=Y-1; % Y signal without afebrille healthy controls


% ADASYN model2 (Sci Rep revision 1: Q2 of R1)
% ADASYN sythetic dataset matchin female sample size with male sample size
% ADASYN: set up ADASYN parameters and call the function:

featuresf = X(Y==1 & X(:,end)==1,1:end-1);
featuresm = X(Y==1 & X(:,end)==0,1:end-1);
labelsf = true([size(featuresf,1) 1]);
labelsm = false([size(featuresm,1) 1]);

adasyn_features                 = [featuresf; featuresm];
adasyn_labels                   = [labelsf  ; labelsm  ];
adasyn_beta                     = [];   %let ADASYN choose default
adasyn_kDensity                 = [];   %let ADASYN choose default
adasyn_kSMOTE                   = [];   %let ADASYN choose default
adasyn_featuresAreNormalized    = false;    %false lets ADASYN handle normalization

% Divide data regarding groups and sex for dataset undersampling and
% testing regression parameter stability (Sci Rep revision 1: Q2 of R2 and Q1 of R1)
XX0f = X(Y==0 & X(:,end)==1,:);
XX0m = X(Y==0 & X(:,end)==0,:);
XX1f = X(Y==1 & X(:,end)==1,:);
XX1m = X(Y==1 & X(:,end)==0,:);
sizeXX0f = size(XX0f,1); sizeXX0f90 = round(0.9*sizeXX0f);  sizeXX0f80 = round(0.8*sizeXX0f);  sizeXX0f70 = round(0.7*sizeXX0f);  sizeXX0f60 = round(0.6*sizeXX0f);  sizeXX0f50 = round(0.5*sizeXX0f);
sizeXX0m = size(XX0m,1); sizeXX0m90 = round(0.9*sizeXX0m);  sizeXX0m80 = round(0.8*sizeXX0m);  sizeXX0m70 = round(0.7*sizeXX0m);  sizeXX0m60 = round(0.6*sizeXX0m);  sizeXX0m50 = round(0.5*sizeXX0m);
sizeXX1f = size(XX1f,1); sizeXX1f90 = round(0.9*sizeXX1f);  sizeXX1f80 = round(0.8*sizeXX1f);  sizeXX1f70 = round(0.7*sizeXX1f);  sizeXX1f60 = round(0.6*sizeXX1f);  sizeXX1f50 = round(0.5*sizeXX1f);
sizeXX1m = size(XX1m,1); sizeXX1m90 = round(0.9*sizeXX1m);  sizeXX1m80 = round(0.8*sizeXX1m);  sizeXX1m70 = round(0.7*sizeXX1m);  sizeXX1m60 = round(0.6*sizeXX1m);  sizeXX1m50 = round(0.5*sizeXX1m);

for iter = 1:niter_adasyn
    % ADASYN
    [adasyn_featuresSyn, adasyn_labelsSyn] = ADASYN(adasyn_features, adasyn_labels, adasyn_beta, adasyn_kDensity, adasyn_kSMOTE, adasyn_featuresAreNormalized);

    XSyn = [X; [adasyn_featuresSyn, adasyn_labelsSyn]];
    YSyn = [Y; ones(size(adasyn_labelsSyn))];

    % Step-wise linear regression for model2
    [m2(iter,1).b,m2(iter,1).se,m2(iter,1).pval,m2(iter,1).inmodel,m2(iter,1).stats,m2(iter,1).nextstep,m2(iter,1).history] = stepwisefit(XSyn,YSyn,'penter',0.05);
    Y_predict = sum( repmat(m2(iter,1).b(m2(iter,1).inmodel==1)',size(X,1),1).*X(:,m2(iter,1).inmodel==1) ,2); % Predicted signal Y_predict
    [r, pr]=corrcoef(Y,Y_predict);
    m2(iter,1).Y_predict = Y_predict;
    m2(iter,1).r = r(1,2);
    m2(iter,1).p_r = pr(1,2);
    m2(iter,1).R2 = (1 - m2(iter,1).stats(1,1).SSresid / m2(iter,1).stats(1,1).SStotal)*100;
    tmp=estimate_threshold(Y_predict(Y==1),Y_predict(Y==0),variable_name,998,'he');
    m2(iter,1).threshold = tmp.OptimThr;
    m2(iter,1).sensitivity = tmp.OptimSensitivity;
    m2(iter,1).specificity = tmp.OptimSpecificity;
    
    % Dataset undersampling and testing of regression coefficient stability (Sci Rep revision 1: Q2 of R2 and Q1 of R1)
    if iter == 1
        m290=regress_undersampled_model2(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f90,sizeXX0m90,sizeXX1f90,sizeXX1m90,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
        m280=regress_undersampled_model2(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f80,sizeXX0m80,sizeXX1f80,sizeXX1m80,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
        m270=regress_undersampled_model2(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f70,sizeXX0m70,sizeXX1f70,sizeXX1m70,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
        m260=regress_undersampled_model2(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f60,sizeXX0m60,sizeXX1f60,sizeXX1m60,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
        m250=regress_undersampled_model2(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f50,sizeXX0m50,sizeXX1f50,sizeXX1m50,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
    else
        m290(iter,1)=regress_undersampled_model2(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f90,sizeXX0m90,sizeXX1f90,sizeXX1m90,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
        m280(iter,1)=regress_undersampled_model2(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f80,sizeXX0m80,sizeXX1f80,sizeXX1m80,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
        m270(iter,1)=regress_undersampled_model2(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f70,sizeXX0m70,sizeXX1f70,sizeXX1m70,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
        m260(iter,1)=regress_undersampled_model2(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f60,sizeXX0m60,sizeXX1f60,sizeXX1m60,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
        m250(iter,1)=regress_undersampled_model2(XX0f,XX0m,XX1f,XX1m,sizeXX0f,sizeXX0m,sizeXX1f,sizeXX1m,sizeXX0f50,sizeXX0m50,sizeXX1f50,sizeXX1m50,adasyn_beta,adasyn_kDensity,adasyn_kSMOTE,adasyn_featuresAreNormalized);
    end
end

M2(1,1).inmodel = reshape([m2(:,1).inmodel]',size(m2(1,1).inmodel,2),niter_adasyn)';
M2(1,1).inmodel_unique = unique(M2.inmodel,'rows');
M2(1,1).inmodel_unique_prob = zeros(size(M2.inmodel_unique,1),1);

for res = 1:size(M2.inmodel_unique,1)
    for iter = 1:niter_adasyn
        M2.inmodel_unique_prob(res,1) = M2.inmodel_unique_prob(res,1) + ...
            double(sum(M2.inmodel_unique(res,:) == M2.inmodel(iter,:)) == size(m2(1,1).inmodel,2));
    end
end
M2(1,1).inmodel_unique_prob = 100*M2.inmodel_unique_prob/niter_adasyn;
M2(1,1).best = find(M2.inmodel_unique_prob==max(M2.inmodel_unique_prob));
M2(1,1).pos = sum(M2.inmodel == M2.inmodel_unique(M2.best,:),2) == size(M2.inmodel,2);
M2(1,1).varid = find(M2.inmodel_unique(M2.best,:)==1);
b = [m2.b]';
b = b(M2.pos,M2.varid);
M2.b(:,1) =  mean(b);
M2.b(:,2) =  std(b);
pval = [m2.pval]';
pval = pval(M2.pos,M2.varid);
M2.pval = median(pval);
r = [m2(M2.pos,1).r]';
M2.r = [mean(r) std(r)];
M2.p_r = median([m2(M2.pos,1).p_r]);
R2 = [m2(M2.pos,1).R2]';
M2.R2 = [mean(R2) std(R2)];
M2.prob = M2.inmodel_unique_prob(M2.best);
M2.unique_models = size(M2.inmodel_unique,1);
th = [m2(M2.pos,1).threshold]';
M2.threshold = [mean(th) std(th)];
th = [m2(M2.pos,1).sensitivity]';
M2.sensitivity = [mean(th) std(th)];
th = [m2(M2.pos,1).specificity]';
M2.specificity = [mean(th) std(th)];
th = arrayfun(@(x) x.stats.rmse,m2); th = th(M2.pos,1);
M2.rmse = [mean(th) std(th)];
th = arrayfun(@(x) x.stats.fstat,m2); th = th(M2.pos,1);
M2.fstat = [mean(th) std(th)];

M290 = get_downsamp_model_info(m290,M2,niter_adasyn);
M280 = get_downsamp_model_info(m280,M2,niter_adasyn);
M270 = get_downsamp_model_info(m270,M2,niter_adasyn);
M260 = get_downsamp_model_info(m260,M2,niter_adasyn);
M250 = get_downsamp_model_info(m250,M2,niter_adasyn);

% Step-wise linear regression for model2
% [model2_b,model2_se,model2_pval,model2_inmodel,model2_stats,model2_nextstep,model2_history] =stepwisefit(X,Y,'penter',0.05);
Y_predict = sum( repmat(M2.b(:,1)',size(X2,1),1).*X2(:,M2.varid) ,2);
Predicted(:,2) = Y_predict;
% [model2_r, model2_pr]=corrcoef(Y,Y_predict(grp_ps~=1));
% model2_R2 = (1 - model2_stats(1,1).SSresid / model2_stats(1,1).SStotal)*100;

% Sorted variables based on variable significance to the model
var_sig=M2.varid;
[~,I] = sort(M2.pval);
var_sig=var_sig(I);
model2_b = M2.b(I);

% Sub-grouping of the predicted signal for betweem-group testings and for visualization purposes 
Yp1=Y_predict(grp_ps==1);
Yp2=Y_predict(grp_ps==2);
Yp3=Y_predict(grp_ps==3 | grp_ps==5);
Yp4=Y_predict(grp_ps==4);
Yp3_1=Y_predict(grp_ps==3);
Yp3_2=Y_predict(grp_ps==5);

% Wilcoxon rank-sum tests
p_Yp24_W(1,1) = ranksum(Yp1,Yp2);
p_Yp24_W(1,2) = ranksum(Yp1,Yp3);
p_Yp24_W(1,3) = ranksum(Yp1,Yp4);
p_Yp24_W(1,4) = ranksum([Yp1; Yp2],[Yp3; Yp4]);
p_Yp24_W(1,5) = ranksum(Yp2,Yp3);
p_Yp24_W(1,6) = ranksum(Yp2,Yp4);
p_Yp24_W(1,7) = ranksum(Yp2,[Yp3; Yp4]);
p_Yp24_W(1,8) = ranksum(Yp3,Yp4);

% Two-sample t-tests
[~, p_Yp24_T(1,1)] = ttest2(Yp1,Yp2);
[~, p_Yp24_T(1,2)] = ttest2(Yp1,Yp3);
[~, p_Yp24_T(1,3)] = ttest2(Yp1,Yp4);
[~, p_Yp24_T(1,4)] = ttest2([Yp1; Yp2],[Yp3; Yp4]);
[~, p_Yp24_T(1,5)] = ttest2(Yp2,Yp3);
[~, p_Yp24_T(1,6)] = ttest2(Yp2,Yp4);
[~, p_Yp24_T(1,7)] = ttest2(Yp2,[Yp3; Yp4]);
[~, p_Yp24_T(1,8)] = ttest2(Yp3,Yp4);

% Sensitivity and specificity analysis
seiz = [Yp3; Yp4];
nonseiz=[Yp2];
sts(1,idx)=estimate_threshold(seiz,nonseiz,variable_name,998,'he');idx=idx+1;
S_risk(2,:)=[sts(1,idx-1).OptimThr sts(1,idx-1).OptimSensitivity sts(1,idx-1).OptimSpecificity];

% Model2 visualization
h(5).fig = figure(5);
set(h(5).fig,'Position',[50 50 1300 500])
subplot(5,2,[1 3 5 7])
scatter3(data(grp==2,var_sig(1)),data(grp==2,var_sig(3)),data(grp==2,var_sig(2)),850, 'g.')
hold on
scatter3(data(grp==1,var_sig(1)),data(grp==1,var_sig(3)),data(grp==1,var_sig(2)),850, 'y.')
scatter3(data(grp==3,var_sig(1)),data(grp==3,var_sig(3)),data(grp==3,var_sig(2)),850, 'b.')
scatter3(data(grp==5,var_sig(1)),data(grp==5,var_sig(3)),data(grp==5,var_sig(2)),100, 'b*','LineWidth',2)
scatter3(data(grp==4,var_sig(1)),data(grp==4,var_sig(3)),data(grp==4,var_sig(2)),850, 'r.')
hold off
grid on
xlabel(variable_name{1,var_sig(1)})
ylabel([variable_name{1,var_sig(3)} ' [\mumol/l]'])
zlabel([variable_name{1,var_sig(2)} ' [%]'])
set(gca,'FontSize',14,...
        'LineWidth',2)

subplot(1,2,2)
if size(var_sig,2)==3
    YLBL2=[num2str(model2_b(1),'%10.3f') '*' variable_name{1,var_sig(1)}  ...
            num2str(model2_b(2),'%10.3f') '*' variable_name{1,var_sig(2)}...
            '+' num2str(model2_b(3),'%10.3f') '*' variable_name{1,var_sig(3)} ];
else
    YLBL2={[num2str(model2_b(1),'%10.3f') '*' variable_name{1,var_sig(1)}  ...
        '+' num2str(model2_b(2),'%10.3f') '*' variable_name{1,var_sig(2)}] ...
        ['+' num2str(model2_b(3),'%10.3f') '*' variable_name{1,var_sig(3)} ...
        num2str(model2_b(4),'%10.3f') '*' variable_name{1,var_sig(4)} ]};
end
plot_cat_scatter(Yp1,Yp2,Yp3_1,Yp3_2,Yp4,YLBL2,'southeast',S_risk(2,:))

subplot(6,2,11)
plot([-2 -1],[0 1])
xlim([0 1])
text(0.4725,0.60,'Sub-group difference testings with Wilcoxon rank-sum test','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','center')
text(0.00,0.25,'1vs2','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.15,0.25,'1vs3','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.30,0.25,'1vs4','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.45,0.25,'12vs34','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.60,0.25,'2vs3','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.75,0.25,'2vs4','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(0.90,0.25,'2vs34','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
text(1.05,0.25,'3vs4','FontSize',fntSiz,'FontWeight','bold','HorizontalAlignment','right')
xp = 0.00:0.15:1.05;
for xi = 1:size(p_Yp24_W,2)
        p = p_Yp24_W(1,xi);
        clr = color_decision(p,thrfwe);
        if p<0.00005
                prec='%10.0e';
        else
                prec='%10.4f';
        end
        if p<thrfwe
                fntwght='bold';
        else
                fntwght='normal';
        end
        text(xp(xi),-0.10,num2str(p,prec),'FontSize',fntSiz,'FontWeight',fntwght,'HorizontalAlignment','right','Color',clr)
end
text(-0.13,-0.45,'Green-bold-highlighted p-values fulfill the corrected significance','FontSize',fntSiz,'FontWeight','normal','HorizontalAlignment','left')
text(-0.13,-0.80,'criterion p_{FWE}<0.05 (Family Wise Error correction).','FontSize',fntSiz,'FontWeight','normal','HorizontalAlignment','left')
axis off

print(fullfile(save_path,'fig2b'), '-dpng', '-r300')
pause(0.2)

%% Multivariate analysis: bi-linear classifier (model1-model2)
% Sensitivity and specificity analysis for bi-linear classifier
thr1_range = sts(1,strcmp({sts.name},'model1')).min:0.005:sts(1,strcmp({sts.name},'model1')).max;
if sts(1,strcmp({sts.name},'model2')).min > 0
    thr2_range = sts(1,strcmp({sts.name},'model2')).min:0.0005:sts(1,strcmp({sts.name},'model2')).max;
else
    thr2_range = 0:0.0005:sts(1,strcmp({sts.name},'model2')).max;
end
[THR1, THR2] = meshgrid(thr1_range,thr2_range);
sts(1,idx).name = 'bi-linear classifier';
for thx = 1:size(THR1,1)
    for thy = 1:size(THR1,2)
        TP=sum(Predicted(:,1)>=THR1(thx,thy) & Predicted(:,2)>=THR2(thx,thy) & (grp_ps==3 | grp_ps==5 | grp_ps==4));
        FP=sum(Predicted(:,1)>=THR1(thx,thy) & Predicted(:,2)>=THR2(thx,thy) & (grp_ps==2));
        TN=sum((Predicted(:,1)<THR1(thx,thy) | Predicted(:,2)<THR2(thx,thy)) & grp_ps==2);
        FN=sum((Predicted(:,1)<THR1(thx,thy) | Predicted(:,2)<THR2(thx,thy)) & (grp_ps==3 | grp_ps==5 | grp_ps==4));
        
        sensitivity=100*TP/(TP+FN);
        specificity=100*TN/(TN+FP);

        sts(1,idx).thr(thx,thy,1) = THR1(thx,thy);
        sts(1,idx).thr(thx,thy,2) = THR2(thx,thy);
        sts(1,idx).TP(thx,thy) = TP;
        sts(1,idx).FN(thx,thy) = FN;
        sts(1,idx).TN(thx,thy) = TN;
        sts(1,idx).FP(thx,thy) = FP;
        sts(1,idx).sensitivity(thx,thy) = sensitivity;
        sts(1,idx).specificity(thx,thy) = specificity;
        sts(1,idx).SensSpecSum(thx,thy) = 1.00*sensitivity+specificity;
    end
end
[sts(1,idx).OptimThrPos(1), sts(1,idx).OptimThrPos(2)] = find(sts(1,idx).SensSpecSum==max(sts(1,idx).SensSpecSum(:)),1,'last');
sts(1,idx).OptimThr = squeeze(sts(1,idx).thr(sts(1,idx).OptimThrPos(1),sts(1,idx).OptimThrPos(2),:))';
sts(1,idx).OptimSensitivity = sts(1,idx).sensitivity(sts(1,idx).OptimThrPos(1),sts(1,idx).OptimThrPos(2));
sts(1,idx).OptimSpecificity = sts(1,idx).specificity(sts(1,idx).OptimThrPos(1),sts(1,idx).OptimThrPos(2));
sts(1,idx).min = [min(thr1_range) min(thr2_range)];
sts(1,idx).max = [max(thr1_range) max(thr2_range)];
S_risk(3,:)=[NaN sts(1,idx).OptimSensitivity sts(1,idx).OptimSpecificity];
S_risk_old = S_risk;
S_risk(1,1)= sts(1,idx).OptimThr(1);
S_risk(2,1)= sts(1,idx).OptimThr(2);
idx=idx+1;

% Visualization of bi-linear classifier
h(6).fig = figure(6);
set(h(6).fig,'Position',[50 50 1300 500])
subplot(1,2,1)
H6=plot([S_risk(1,1) S_risk(1,1)],[-5 5],'k','LineWidth',2);
hold on
plot([-5 5],[S_risk(2,1) S_risk(2,1)],'k','LineWidth',2)
H1=scatter(Predicted(grp_ps==1,1),Predicted(grp_ps==1,2),850, 'y.','MarkerEdgeAlpha',0.5);
H2=scatter(Predicted(grp_ps==2,1),Predicted(grp_ps==2,2),850, 'g.','MarkerEdgeAlpha',0.5);
H3=scatter(Predicted(grp_ps==3,1),Predicted(grp_ps==3,2),850, 'b.','MarkerEdgeAlpha',0.5);
H4=scatter(Predicted(grp_ps==5,1),Predicted(grp_ps==5,2),100, 'b*','MarkerEdgeAlpha',0.5,'LineWidth',2);
H5=scatter(Predicted(grp_ps==4,1),Predicted(grp_ps==4,2),850, 'r.','MarkerEdgeAlpha',0.5);
% H7=plot([-1000 -1000],[-1000 -1000],'c-.','LineWidth',2);
hold off
grid on
axis([ 1.02*min(Predicted(:,1)) fig6xmax 1.1*min(Predicted(:,2)) fig6ymax ])
title('Data in space of two regressed variables')
xlabel(YLBL1)
ylabel(YLBL2)
legend([H1 H2 H3 H4 H5 H6],{'healthy controls','without seizures','non-recurrent seizures',...
        'non-r. complex seizures','recurrent seizures','bi-linear classifier'},'location','southeast')
set(gca,'FontSize',13,'LineWidth',2)

subplot(1,2,2)
plot([S_risk(1,1) S_risk(1,1)],[-5 5],'k','LineWidth',2)
hold on
plot([-5 5],[S_risk(2,1) S_risk(2,1)],'k','LineWidth',2)
scatter(Predicted(grp_ps==1,1),Predicted(grp_ps==1,2),850, 'y.','MarkerEdgeAlpha',0.5)
scatter(Predicted(grp_ps==2,1),Predicted(grp_ps==2,2),850, 'g.','MarkerEdgeAlpha',0.5)
scatter(Predicted(grp_ps==3,1),Predicted(grp_ps==3,2),850, 'b.','MarkerEdgeAlpha',0.5)
scatter(Predicted(grp_ps==5,1),Predicted(grp_ps==5,2),100, 'b*','MarkerEdgeAlpha',0.5,'LineWidth',2)
scatter(Predicted(grp_ps==4,1),Predicted(grp_ps==4,2),850, 'r.','MarkerEdgeAlpha',0.5)
text(0.75,0.18,'Bi-linear classifier:','FontSize',12)
text(0.75,0.13,['SE=' num2str(S_risk(3,2),'%10.1f') '%; SP=' num2str(S_risk(3,3),'%10.1f') '%'],'FontSize',12)
hold off
grid on
axis([ fig6xmin_zoom fig6xmax 0 fig6ymax ])
title('Detail at the cut-off area')
xlabel(YLBL1)
ylabel(YLBL2)
set(gca,'FontSize',13,'LineWidth',2)
print(fullfile(save_path,'fig3a'), '-dpng', '-r300')
pause(0.2)

%% Receiver operating characteristic: visualization
blnr_class=unique([sts(1,strcmp({sts.name},'bi-linear classifier')).sensitivity(:) sts(1,strcmp({sts.name},'bi-linear classifier')).specificity(:)],'rows');
distances = sqrt((0-(100-[sts(1,:).OptimSpecificity])).^2 + (100-[sts(1,:).OptimSensitivity]).^2);
h(7).fig = figure(7);
set(h(7).fig,'Position',[50 50 1300 500])
subplot(1,2,1)
H6=scatter(100-blnr_class(:,2),blnr_class(:,1),28,'x','LineWidth',2,'MarkerEdgeColor',[1 0 0]);
hold on
H4=plot(100-sts(1,strcmp({sts.name},'model1')).specificity,sts(1,strcmp({sts.name},'model1')).sensitivity,'LineWidth',5,'Color',[1 1 0]);
H5=plot(100-sts(1,strcmp({sts.name},'model2')).specificity,sts(1,strcmp({sts.name},'model2')).sensitivity,'LineWidth',4,'Color','c');
H3=plot(100-sts(1,strcmp({sts.name},'UIBC')).specificity,sts(1,strcmp({sts.name},'UIBC')).sensitivity,'LineWidth',3,'Color',[0 0 1]);
H1=plot(100-sts(1,strcmp({sts.name},'satFe')).specificity,sts(1,strcmp({sts.name},'satFe')).sensitivity,'LineWidth',3,'Color',[0.5 0.5 0.5]);
H2=plot(100-sts(1,strcmp({sts.name},'Fe')).specificity,sts(1,strcmp({sts.name},'Fe')).sensitivity,'LineWidth',3,'Color',[0 1 0]);
scatter(100-sts(1,strcmp({sts.name},'model1')).OptimSpecificity,sts(1,strcmp({sts.name},'model1')).OptimSensitivity,120,'o','LineWidth',2,'MarkerEdgeColor',[0 0 0],'MarkerFaceColor',[1 1 0])
scatter(100-sts(1,strcmp({sts.name},'model2')).OptimSpecificity,sts(1,strcmp({sts.name},'model2')).OptimSensitivity,120,'o','LineWidth',2,'MarkerEdgeColor',[0 0 0],'MarkerFaceColor','c')
scatter(100-sts(1,strcmp({sts.name},'UIBC')).OptimSpecificity,sts(1,strcmp({sts.name},'UIBC')).OptimSensitivity,120,'o','LineWidth',2,'MarkerEdgeColor',[0 0 0],'MarkerFaceColor',[0 0 1])
scatter(100-sts(1,strcmp({sts.name},'satFe')).OptimSpecificity,sts(1,strcmp({sts.name},'satFe')).OptimSensitivity,120,'o','LineWidth',2,'MarkerEdgeColor',[0 0 0],'MarkerFaceColor',[0.5 0.5 0.5])
scatter(100-sts(1,strcmp({sts.name},'Fe')).OptimSpecificity,sts(1,strcmp({sts.name},'Fe')).OptimSensitivity,120,'o','LineWidth',2,'MarkerEdgeColor',[0 0 0],'MarkerFaceColor',[0 1 0])
scatter(100-sts(1,strcmp({sts.name},'bi-linear classifier')).OptimSpecificity,sts(1,strcmp({sts.name},'bi-linear classifier')).OptimSensitivity,120,'o','LineWidth',2,'MarkerEdgeColor',[0 0 0],'MarkerFaceColor',[1 0 0])
H7=scatter(-1,-1,120,'o','LineWidth',2,'MarkerEdgeColor',[0 0 0],'MarkerFaceColor',[1 1 1]);
H8=scatter(0,100,120,'o','LineWidth',2,'MarkerEdgeColor',[255,215,0]/255,'MarkerFaceColor',[255,215,0]/255);
hold off
legend([H1 H2 H3 H4 H5 H6 H7 H8],{[sts(1,strcmp({sts.name},'satFe')).name ': E=' num2str(distances(1,strcmp({sts.name},'satFe')),'%10.1f')], ...
    [sts(1,strcmp({sts.name},'Fe')).name ': E=' num2str(distances(1,strcmp({sts.name},'Fe')),'%10.1f')],...
    [sts(1,strcmp({sts.name},'UIBC')).name ': E=' num2str(distances(1,strcmp({sts.name},'UIBC')),'%10.1f')], ...
    [sts(1,strcmp({sts.name},'model1')).name ': E=' num2str(distances(1,strcmp({sts.name},'model1')),'%10.1f')], ...
    [sts(1,strcmp({sts.name},'model2')).name ': E=' num2str(distances(1,strcmp({sts.name},'model2')),'%10.1f')], ...
    [sts(1,strcmp({sts.name},'bi-linear classifier')).name ': E=' num2str(distances(1,strcmp({sts.name},'bi-linear classifier')),'%10.1f')],...
    'the most optimal classifier fit','ideal classifier - i.e. gold standard'},'Location','SouthEast')
xlabel('100-specificity [%]')
ylabel('sensitivity [%]')
title('Receiver operating characteristic')
axis([0 100 0 100])
grid on
set(gca,'FontSize',13,'LineWidth',2)
print(fullfile(save_path,'fig3b'), '-dpng', '-r300')
pause(0.2)

%% Build table with descriptive statistics of demographical variables and iron status
tbl{1,2} = 'Units';
tbl{1,3} = 'FS';
tbl{2,3} = 'Median';
tbl{2,4} = 'Mean';
tbl{2,5} = 'SD';
tbl{1,6} = 'Non-recurrent FS';
tbl{2,6} = 'Median';
tbl{2,7} = 'Mean';
tbl{2,8} = 'SD';
tbl{1,9} = 'Recurrent FS';
tbl{2,9} = 'Median';
tbl{2,10} = 'Mean';
tbl{2,11} = 'SD';
tbl{1,12} = 'Febrile controls';
tbl{2,12} = 'Median';
tbl{2,13} = 'Mean';
tbl{2,14} = 'SD';
tbl{1,15} = 'Healthy controls';
tbl{2,15} = 'Median';
tbl{2,16} = 'Mean';
tbl{2,17} = 'SD';

tbl{3,1} = 'Demographics';
tbl{4,1} = 'Sex';tbl{4,2} = 'females/%';
tbl{5,1} = 'GA';tbl{5,2} = 'weeks';
tbl{6,1} = 'Age';tbl{6,2} = 'months';
tbl{7,1} = 'Height';tbl{7,2} = 'percentile';
tbl{8,1} = 'Weight';tbl{8,2} = 'percentile';
tbl{9,1} = 'Age at 1st FS';tbl{9,2} = 'months';
tbl{10,1} = 'Temperature';tbl{10,2} = '°C';
tbl{11,1} = 'FS duration';tbl{11,2} = 'min';

tbl{12,1} = 'Iron status';
tbl{13,1} = 'RBC';tbl{13,2} = '10e06/\muL'; %Ery
tbl{14,1} = 'HGB';tbl{14,2} = 'g/L';
tbl{15,1} = 'Fe';tbl{15,2} = '\mumol/L';
tbl{16,1} = 'Fer';tbl{16,2} = 'ng/mL';
tbl{17,1} = 'TF';tbl{17,2} = 'g/L';
tbl{18,1} = 'satFe';tbl{18,2} = '%';
tbl{19,1} = 'UIBC';tbl{19,2} = '\mumol/L';

tbl{3,3} = sum(grp>=3 & firstrecord==1);
tbl{3,6} = sum((grp==3 | grp==5) & firstrecord==1);
tbl{3,9} = sum(grp==4 & firstrecord==1);
tbl{3,12} = sum(grp==2 & firstrecord==1);
tbl{3,15} = sum(grp==1 & firstrecord==1);

tbl{4,3} = sum(female==1 & grp>=3 & firstrecord==1);
tbl{4,4} = sum(female==1 & grp>=3 & firstrecord==1)*100/sum(grp>=3 & firstrecord==1);
tbl{4,6} = sum(female==1 & (grp==3 | grp==5) & firstrecord==1);
tbl{4,7} = sum(female==1 & (grp==3 | grp==5) & firstrecord==1)*100/sum((grp==3 | grp==5) & firstrecord==1);
tbl{4,9} = sum(female==1 & grp==4 & firstrecord==1);
tbl{4,10} = sum(female==1 & grp==4 & firstrecord==1)*100/sum(grp==4 & firstrecord==1);
tbl{4,12} = sum(female==1 & grp==2 & firstrecord==1);
tbl{4,13} = sum(female==1 & grp==2 & firstrecord==1)*100/sum(grp==2 & firstrecord==1);
tbl{4,15} = sum(female==1 & grp==1 & firstrecord==1);
tbl{4,16} = sum(female==1 & grp==1 & firstrecord==1)*100/sum(grp==1 & firstrecord==1);

posdata = [1 2 3 4 5 6 7 8 9 10]; % Positions in data matrix
postbl = [5 6 7 8 14 15 16 17 18 19]; % Positions in table for the smae variables
for ind = 1:size(posdata,2)
    tbl{postbl(ind),3} = nanmedian(data(grp>=3,posdata(ind)));
    tbl{postbl(ind),4} = nanmean(data(grp>=3,posdata(ind)));
    tbl{postbl(ind),5} = nanstd(data(grp>=3,posdata(ind)));
    tbl{postbl(ind),6} = nanmedian(data(grp==3 | grp==5,posdata(ind)));
    tbl{postbl(ind),7} = nanmean(data(grp==3 | grp==5,posdata(ind)));
    tbl{postbl(ind),8} = nanstd(data(grp==3 | grp==5,posdata(ind)));
    tbl{postbl(ind),9} = nanmedian(data(grp==4,posdata(ind)));
    tbl{postbl(ind),10} = nanmean(data(grp==4,posdata(ind)));
    tbl{postbl(ind),11} = nanstd(data(grp==4,posdata(ind)));
    tbl{postbl(ind),12} = nanmedian(data(grp==2,posdata(ind)));
    tbl{postbl(ind),13} = nanmean(data(grp==2,posdata(ind)));
    tbl{postbl(ind),14} = nanstd(data(grp==2,posdata(ind)));
    tbl{postbl(ind),15} = nanmedian(data(grp==1,posdata(ind)));
    tbl{postbl(ind),16} = nanmean(data(grp==1,posdata(ind)));
    tbl{postbl(ind),17} = nanstd(data(grp==1,posdata(ind)));
end

tbl{9,3} = nanmedian(age1stfs(grp>=3 & firstrecord==1));
tbl{9,4} = nanmean(age1stfs(grp>=3 & firstrecord==1));
tbl{9,5} = nanstd(age1stfs(grp>=3 & firstrecord==1));
tbl{9,6} = nanmedian(age1stfs((grp==3 |grp==5) & firstrecord==1));
tbl{9,7} = nanmean(age1stfs((grp==3 |grp==5) & firstrecord==1));
tbl{9,8} = nanstd(age1stfs((grp==3 |grp==5) & firstrecord==1));
tbl{9,9} = nanmedian(age1stfs(grp==4 & firstrecord==1));
tbl{9,10} = nanmean(age1stfs(grp==4 & firstrecord==1));
tbl{9,11} = nanstd(age1stfs(grp==4 & firstrecord==1));

tbl{10,3} = nanmedian(temperature(grp>=3));
tbl{10,4} = nanmean(temperature(grp>=3));
tbl{10,5} = nanstd(temperature(grp>=3));
tbl{10,6} = nanmedian(temperature(grp==3 |grp==5));
tbl{10,7} = nanmean(temperature(grp==3 |grp==5));
tbl{10,8} = nanstd(temperature(grp==3 |grp==5));
tbl{10,9} = nanmedian(temperature(grp==4));
tbl{10,10} = nanmean(temperature(grp==4));
tbl{10,11} = nanstd(temperature(grp==4));
tbl{10,12} = nanmedian(temperature(grp==2));
tbl{10,13} = nanmean(temperature(grp==2));
tbl{10,14} = nanstd(temperature(grp==2));

tbl{11,3} = nanmedian(seize_duration(grp>=3));
tbl{11,4} = nanmean(seize_duration(grp>=3));
tbl{11,5} = nanstd(seize_duration(grp>=3));
tbl{11,6} = nanmedian(seize_duration(grp==3 |grp==5));
tbl{11,7} = nanmean(seize_duration(grp==3 |grp==5));
tbl{11,8} = nanstd(seize_duration(grp==3 |grp==5));
tbl{11,9} = nanmedian(seize_duration(grp==4));
tbl{11,10} = nanmean(seize_duration(grp==4));
tbl{11,11} = nanstd(seize_duration(grp==4));

tbl{13,3} = nanmedian(rbc(grp>=3));
tbl{13,4} = nanmean(rbc(grp>=3));
tbl{13,5} = nanstd(rbc(grp>=3));
tbl{13,6} = nanmedian(rbc(grp==3 |grp==5));
tbl{13,7} = nanmean(rbc(grp==3 |grp==5));
tbl{13,8} = nanstd(rbc(grp==3 |grp==5));
tbl{13,9} = nanmedian(rbc(grp==4));
tbl{13,10} = nanmean(rbc(grp==4));
tbl{13,11} = nanstd(rbc(grp==4));
tbl{13,12} = nanmedian(rbc(grp==2));
tbl{13,13} = nanmean(rbc(grp==2));
tbl{13,14} = nanstd(rbc(grp==2));
tbl{13,15} = nanmedian(rbc(grp==1));
tbl{13,16} = nanmean(rbc(grp==1));
tbl{13,17} = nanstd(rbc(grp==1));


for ind = 3:size(tbl,2)
    tbl{strcmp(tbl(:,1),'satFe'),ind}=tbl{strcmp(tbl(:,1),'satFe'),ind}*100;
end

%% Analysis of Sex impact on investigated variables in data matrix
% tblsex cell array consistf of p-values of Wilcoxon-rank sum tests
% tblsex2 cell array consistf of p-values of two-sample t-tests
% sx1m_stat consists of male descriptive statistics for healthy controls
% sx1f_stat consists of female descriptive statistics for healthy controls
tblsex{1,1} = 'Variable';
tblsex{1,2} = 1;
tblsex{1,3} = 2;
tblsex{1,4} = 3;
tblsex{1,5} = 4;
tblsex{1,6} = '34';

tblsex2 = tblsex;
sx1m_stat = zeros(size(data,2)+1,3);
sx1f_stat = zeros(size(data,2)+1,3);

for ind = 1:size(data,2)
    sx1m = data(grp==1 & female==0,ind);
    sx1f = data(grp==1 & female==1,ind);
    sx2m = data(grp==2 & female==0,ind);
    sx2f = data(grp==2 & female==1,ind);
    sx3m = data((grp==3 | grp==5) & female==0,ind);
    sx3f = data((grp==3 | grp==5) & female==1,ind);
    sx4m = data(grp==4 & female==0,ind);
    sx4f = data(grp==4 & female==1,ind);
    sx34m = data((grp==3 | grp==5  | grp==4) & female==0,ind);
    sx34f = data((grp==3 | grp==5  | grp==4) & female==1,ind);
    
    sx1m = sx1m(~isnan(sx1m));
    sx1f = sx1f(~isnan(sx1f));
    sx2m = sx2m(~isnan(sx2m));
    sx2f = sx2f(~isnan(sx2f));
    sx3m = sx3m(~isnan(sx3m));
    sx3f = sx3f(~isnan(sx3f));
    sx4m = sx4m(~isnan(sx4m));
    sx4f = sx4f(~isnan(sx4f));
    sx34m = sx34m(~isnan(sx34m));
    sx34f = sx34f(~isnan(sx34f));
    
    sx1m_stat(ind+1,:) = quantile(sx1m,[.25 .50 .75]);
    sx1f_stat(ind+1,:) = quantile(sx1f,[.25 .50 .75]);
    
    tblsex{ind+1,2} = ranksum(sx1m,sx1f);
    tblsex{ind+1,3} = ranksum(sx2m,sx2f);
    tblsex{ind+1,4} = ranksum(sx3m,sx3f);
    tblsex{ind+1,5} = ranksum(sx4m,sx4f);
    tblsex{ind+1,6} = ranksum(sx34m,sx34f);
    
    tblsex{ind+1,1} = variable_name{1,ind};
    
    [~, tblsex2{ind+1,2}] = ttest2(sx1m,sx1f);
    [~, tblsex2{ind+1,3}] = ttest2(sx2m,sx2f);
    [~, tblsex2{ind+1,4}] = ttest2(sx3m,sx3f);
    [~, tblsex2{ind+1,5}] = ttest2(sx4m,sx4f);
    [~, tblsex2{ind+1,6}] = ttest2(sx34m,sx34f);
    
    tblsex2{ind+1,1} = variable_name{1,ind};
end
%% Build table of model sensitivity to dataset downsampling
tblsampl{1,2} = 'Dataset size';
tblsampl{1,3} = '100%'; tblsampl{1,5} = '90%';  tblsampl{1,7} = '80%'; tblsampl{1,9} = '70%'; tblsampl{1,11} = '60%'; tblsampl{1,13} = '50%';
tblsampl{2,3} = 'Mean';tblsampl{2,4} = 'STD'; tblsampl{2,5} = 'Mean';tblsampl{2,6} = 'STD'; tblsampl{2,7} = 'Mean';tblsampl{2,8} = 'STD'; tblsampl{2,9} = 'Mean';tblsampl{2,10} = 'STD'; tblsampl{2,11} = 'Mean';tblsampl{2,12} = 'STD'; tblsampl{2,13} = 'Mean';tblsampl{2,14} = 'STD';

tblsampl{3,1} = 'Model1';
tblsampl{3,2} = 'Model detection rate [%]';
tblsampl{3,3} = M1.prob; tblsampl{3,5} = M190.prob;  tblsampl{3,7} = M180.prob;  tblsampl{3,9} = M170.prob;  tblsampl{3,11} = M160.prob;   tblsampl{3,13} = M150.prob;
tblsampl{4,2} = 'Total number of identified models';
tblsampl{4,3} = M1.unique_models; tblsampl{4,5} = M190.unique_models;  tblsampl{4,7} = M180.unique_models;  tblsampl{4,9} = M170.unique_models;  tblsampl{4,11} = M160.unique_models;   tblsampl{4,13} = M150.unique_models;
for mdx1 = 1:size(M1.varid,2)
    tblsampl{4+mdx1,2} = [variable_name{1,M1.varid(mdx1)} ' regression coefficient'];
    b = [M1.b(mdx1,:) M190.b(mdx1,:) M180.b(mdx1,:) M170.b(mdx1,:) M160.b(mdx1,:) M150.b(mdx1,:)];
    for pos=1:size(b,2)
        tblsampl{4+mdx1,2+pos} = b(pos);
    end
end
tblsampl{4+mdx1+1,2} = 'F-statistics';
tblsampl{4+mdx1+2,2} = 'Root mean square error';
tblsampl{4+mdx1+3,2} = 'Explained variance R^2 [%]';
tblsampl{4+mdx1+4,2} = 'Pearson correlation (y vs y_predicted)';
tblsampl{4+mdx1+5,2} = 'Non-seizure/seizure separating threshold';
tblsampl{4+mdx1+6,2} = 'Training: sensitivity';
tblsampl{4+mdx1+7,2} = 'Training: specificity';
tblsampl{4+mdx1+8,2} = 'Testing: sensitivity';
tblsampl{4+mdx1+9,2} = 'Testing: specificity';
fstat = [M1.fstat M190.fstat M180.fstat M170.fstat M160.fstat M150.fstat];
rmse = [M1.rmse M190.rmse M180.rmse M170.rmse M160.rmse M150.rmse];
testSE = [NaN NaN M190.testSE M180.testSE M170.testSE M160.testSE M150.testSE];
testSP = [NaN NaN M190.testSP M180.testSP M170.testSP M160.testSP M150.testSP];
R2 = [M1.R2 M190.R2 M180.R2 M170.R2 M160.R2 M150.R2];
r = [M1.r M190.r M180.r M170.r M160.r M150.r];
th = [M1.threshold M190.threshold M180.threshold M170.threshold M160.threshold M150.threshold];
sensitivity = [M1.sensitivity M190.sensitivity M180.sensitivity M170.sensitivity M160.sensitivity M150.sensitivity];
specificity = [M1.specificity M190.specificity M180.specificity M170.specificity M160.specificity M150.specificity];
for pos=1:size(R2,2)
    tblsampl{4+mdx1+1,2+pos} = fstat(pos);
    tblsampl{4+mdx1+2,2+pos} = rmse(pos);
    tblsampl{4+mdx1+3,2+pos} = R2(pos);
    tblsampl{4+mdx1+4,2+pos} = r(pos);
    tblsampl{4+mdx1+5,2+pos} = th(pos);
    tblsampl{4+mdx1+6,2+pos} = sensitivity(pos);
    tblsampl{4+mdx1+7,2+pos} = specificity(pos);
    if ~isnan(testSE(pos))
        tblsampl{4+mdx1+8,2+pos} = testSE(pos);
        tblsampl{4+mdx1+9,2+pos} = testSP(pos);
    end
end


tblsampl{4+mdx1+10,1} = 'Model2';
tblsampl{4+mdx1+10,2} = 'Model detection rate [%]';
tblsampl{4+mdx1+10,3} = M2.prob; tblsampl{4+mdx1+10,5} = M290.prob;  tblsampl{4+mdx1+10,7} = M280.prob;  tblsampl{4+mdx1+10,9} = M270.prob;  tblsampl{4+mdx1+10,11} = M260.prob;   tblsampl{4+mdx1+10,13} = M250.prob;
tblsampl{4+mdx1+11,2} = 'Total number of identified models';
tblsampl{4+mdx1+11,3} = M2.unique_models; tblsampl{4+mdx1+11,5} = M290.unique_models; tblsampl{4+mdx1+11,7} = M280.unique_models; tblsampl{4+mdx1+11,9} = M270.unique_models;  tblsampl{4+mdx1+11,11} = M260.unique_models;   tblsampl{4+mdx1+11,13} = M250.unique_models;
for mdx2 = 1:size(M2.varid,2)
    tblsampl{4+mdx1+11+mdx2,2} = [variable_name{1,M2.varid(mdx2)} ' regression coefficient'];
    b = [M2.b(mdx2,:) M290.b(mdx2,:) M280.b(mdx2,:) M270.b(mdx2,:) M260.b(mdx2,:) M250.b(mdx2,:)];
    for pos=1:size(b,2)
        tblsampl{4+mdx1+11+mdx2,2+pos} = b(pos);
    end
end
tblsampl{4+mdx1+11+mdx2+1,2} = 'F-statistics';
tblsampl{4+mdx1+11+mdx2+2,2} = 'Root mean square error';
tblsampl{4+mdx1+11+mdx2+3,2} = 'Explained variance R^2 [%]';
tblsampl{4+mdx1+11+mdx2+4,2} = 'Pearson correlation (y vs y_predicted)';
tblsampl{4+mdx1+11+mdx2+5,2} = 'Non-seizure/seizure separating threshold';
tblsampl{4+mdx1+11+mdx2+6,2} = 'Training: sensitivity';
tblsampl{4+mdx1+11+mdx2+7,2} = 'Training: specificity';
tblsampl{4+mdx1+11+mdx2+8,2} = 'Testing: sensitivity';
tblsampl{4+mdx1+11+mdx2+9,2} = 'Testing: specificity';
fstat = [M2.fstat M290.fstat M280.fstat M270.fstat M260.fstat M250.fstat];
rmse = [M2.rmse M290.rmse M280.rmse M270.rmse M260.rmse M250.rmse];
testSE = [NaN NaN M290.testSE M280.testSE M270.testSE M260.testSE M250.testSE];
testSP = [NaN NaN M290.testSP M280.testSP M270.testSP M260.testSP M250.testSP];
R2 = [M2.R2 M290.R2 M280.R2 M270.R2 M260.R2 M250.R2];
r = [M2.r M290.r M280.r M270.r M260.r M250.r];
th = [M2.threshold M290.threshold M280.threshold M270.threshold M260.threshold M250.threshold];
sensitivity = [M2.sensitivity M290.sensitivity M280.sensitivity M270.sensitivity M260.sensitivity M250.sensitivity];
specificity = [M2.specificity M290.specificity M280.specificity M270.specificity M260.specificity M250.specificity];
for pos=1:size(R2,2)
    tblsampl{4+mdx1+11+mdx2+1,2+pos} = fstat(pos);
    tblsampl{4+mdx1+11+mdx2+2,2+pos} = rmse(pos);
    tblsampl{4+mdx1+11+mdx2+3,2+pos} = R2(pos);
    tblsampl{4+mdx1+11+mdx2+4,2+pos} = r(pos);
    tblsampl{4+mdx1+11+mdx2+5,2+pos} = th(pos);
    tblsampl{4+mdx1+11+mdx2+6,2+pos} = sensitivity(pos);
    tblsampl{4+mdx1+11+mdx2+7,2+pos} = specificity(pos);
    if ~isnan(testSE(pos))
        tblsampl{4+mdx1+11+mdx2+8,2+pos} = testSE(pos);
        tblsampl{4+mdx1+11+mdx2+9,2+pos} = testSP(pos);
    end
end


tblsampl{4+mdx1+11+mdx2+10,1} = 'Model3';
tblsampl{4+mdx1+11+mdx2+10,2} = 'Model detection rate [%]';
tblsampl{4+mdx1+11+mdx2+10,3} = M3.prob; tblsampl{4+mdx1+11+mdx2+10,5} = M390.prob;  tblsampl{4+mdx1+11+mdx2+10,7} = M380.prob;  tblsampl{4+mdx1+11+mdx2+10,9} = M370.prob;  tblsampl{4+mdx1+11+mdx2+10,11} = M360.prob;   tblsampl{4+mdx1+11+mdx2+10,13} = M350.prob;
tblsampl{4+mdx1+11+mdx2+11,2} = 'Total number of identified models';
tblsampl{4+mdx1+11+mdx2+11,3} = M3.unique_models; tblsampl{4+mdx1+11+mdx2+11,5} = M390.unique_models;  tblsampl{4+mdx1+11+mdx2+11,7} = M380.unique_models;  tblsampl{4+mdx1+11+mdx2+11,9} = M370.unique_models;  tblsampl{4+mdx1+11+mdx2+11,11} = M360.unique_models;   tblsampl{4+mdx1+11+mdx2+11,13} = M350.unique_models;
for mdx3 = 1:size(M3.varid,2)
    tblsampl{4+mdx1+11+mdx2+11+mdx3,2} = [variable_name{1,M3.varid(mdx3)} ' regression coefficient'];
    b = [M3.b(mdx3,:) M390.b(mdx3,:) M380.b(mdx3,:) M370.b(mdx3,:) M360.b(mdx3,:) M350.b(mdx3,:)];
    for pos=1:size(b,2)
        tblsampl{4+mdx1+11+mdx2+11+mdx3,2+pos} = b(pos);
    end
end
tblsampl{4+mdx1+11+mdx2+11+mdx3+1,2} = 'F-statistics';
tblsampl{4+mdx1+11+mdx2+11+mdx3+2,2} = 'Root mean square error';
tblsampl{4+mdx1+11+mdx2+11+mdx3+3,2} = 'Explained variance R^2 [%]';
tblsampl{4+mdx1+11+mdx2+11+mdx3+4,2} = 'Pearson correlation (y vs y_predicted)';
tblsampl{4+mdx1+11+mdx2+11+mdx3+5,2} = 'Non-recurrent/recurrent seizure separating threshold';
tblsampl{4+mdx1+11+mdx2+11+mdx3+6,2} = 'Training: sensitivity';
tblsampl{4+mdx1+11+mdx2+11+mdx3+7,2} = 'Training: specificity';
tblsampl{4+mdx1+11+mdx2+11+mdx3+8,2} = 'Testing: sensitivity';
tblsampl{4+mdx1+11+mdx2+11+mdx3+9,2} = 'Testing: specificity';
fstat = [M3.fstat M390.fstat M380.fstat M370.fstat M360.fstat M350.fstat];
rmse = [M3.rmse M390.rmse M380.rmse M370.rmse M360.rmse M350.rmse];
testSE = [NaN NaN M390.testSE M380.testSE M370.testSE M360.testSE M350.testSE];
testSP = [NaN NaN M390.testSP M380.testSP M370.testSP M360.testSP M350.testSP];
R2 = [M3.R2 M390.R2 M380.R2 M370.R2 M360.R2 M350.R2];
r = [M3.r M390.r M380.r M370.r M360.r M350.r];
th = [M3.threshold M390.threshold M380.threshold M370.threshold M360.threshold M350.threshold];
sensitivity = [M3.sensitivity M390.sensitivity M380.sensitivity M370.sensitivity M360.sensitivity M350.sensitivity];
specificity = [M3.specificity M390.specificity M380.specificity M370.specificity M360.specificity M350.specificity];
for pos=1:size(R2,2)
    tblsampl{4+mdx1+11+mdx2+11+mdx3+1,2+pos} = fstat(pos);
    tblsampl{4+mdx1+11+mdx2+11+mdx3+2,2+pos} = rmse(pos);
    tblsampl{4+mdx1+11+mdx2+11+mdx3+3,2+pos} = R2(pos);
    tblsampl{4+mdx1+11+mdx2+11+mdx3+4,2+pos} = r(pos);
    tblsampl{4+mdx1+11+mdx2+11+mdx3+5,2+pos} = th(pos);
    tblsampl{4+mdx1+11+mdx2+11+mdx3+6,2+pos} = sensitivity(pos);
    tblsampl{4+mdx1+11+mdx2+11+mdx3+7,2+pos} = specificity(pos);
    if ~isnan(testSE(pos))
        tblsampl{4+mdx1+11+mdx2+11+mdx3+8,2+pos} = testSE(pos);
        tblsampl{4+mdx1+11+mdx2+11+mdx3+9,2+pos} = testSP(pos);
    end
end