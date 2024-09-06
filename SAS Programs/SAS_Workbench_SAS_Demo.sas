
/*********************************************/
/* SAS Workbench Machine Learning Procedures */
/*********************************************/

libname mydata "/workspaces/myfolder/mydata/";

data work.hmeq;
    set mydata.hmeq;
run;

/*************/
/* View Data */
/*************/

proc contents data=hmeq;
run;

title "View Table";
proc print data=hmeq (obs=5);
run;
title;

title "Frequency Tables for Nominal Variables";
proc freq data=hmeq;
    table BAD JOB REASON;
run;
title;

title "Descriptive Statistics for Interval Variables";
proc means data=hmeq;
    var LOAN MORTDUE VALUE YOJ DEROG DELINQ CLAGE NINQ CLNO DEBTINC;
run;
title;

title "Histograms For Interval Variables";
ods select histogram;
proc univariate data=hmeq;
histogram LOAN MORTDUE VALUE YOJ DEROG DELINQ CLAGE NINQ CLNO DEBTINC / normal(mu=est sigma=est);
run;
title;

title "Correlation Matrix For Interval Variables";
ods select pearsoncorr;
proc corr data=hmeq noprob nomiss;
    var LOAN MORTDUE VALUE YOJ DEROG DELINQ CLAGE NINQ CLNO DEBTINC;
run;
title;

title "Scatter Plot For The Two Variables With Largest Correlation";
proc sgplot data=hmeq;
    scatter x=VALUE y=MORTDUE / group=BAD;
run;
title;

title "Chi-Square Test Of Independence For Target and Nominal Variables";
proc freq data=hmeq;
    table BAD*JOB / chisq;
    table BAD*REASON / chisq;
run;
title;

/******************/
/* Partition Data */
/******************/

title "Partition Data Into 70% Training and 30% Validation";
proc partition data=hmeq seed=802 samppct=70 partind;
    by BAD;
    output out=hmeq;
run;
title;

title "Observations By Partition";
proc freq data=hmeq;
    table _PARTIND_;
run;
title;

/******************/
/* Impute Missing */
/******************/

title "Number Observations Missing For Interval Variables";
proc means data=hmeq nmiss;
    var LOAN MORTDUE VALUE YOJ DEROG DELINQ CLAGE NINQ CLNO DEBTINC;
run;
title;

title "Number Observations Missing For Nominal Variables";
proc sql; 
    select nmiss(BAD) as BAD_miss, nmiss(JOB) as JOB_miss, nmiss(REASON) as REASON_miss
    from hmeq; 
quit;
title;

title "Imputation Code From Training Data";
proc varimpute data=hmeq (where=(_PARTIND_=1)) seed=802;
    input JOB REASON / ntech=mode;
    input LOAN MORTDUE VALUE YOJ DEROG DELINQ CLAGE NINQ CLNO DEBTINC / ctech=median;
    code file="/workspaces/myfolder/mystuff/impute_score_code.sas";
run;
title;

title "Impute All Data";
data hmeq;
    set hmeq;
    %include "/workspaces/myfolder/mystuff/impute_score_code.sas";
run;
title;

title "View New Imputation Variables";
proc print data=hmeq (obs=5);
    var _PARTIND_ BAD IM_JOB IM_REASON IM_LOAN IM_MORTDUE IM_VALUE 
    IM_YOJ IM_DEROG IM_DELINQ IM_CLAGE IM_NINQ IM_CLNO IM_DEBTINC;
run;
title;

/***********************/
/* Dimension Reduction */
/***********************/

proc varreduce data=hmeq;
    class IM_JOB IM_REASON;
    reduce unsupervised IM_JOB IM_REASON IM_LOAN IM_MORTDUE IM_VALUE 
    IM_YOJ IM_DEROG IM_DELINQ IM_CLAGE IM_NINQ IM_CLNO IM_DEBTINC / varexp=0.95;
    ods output selectedeffects=selectedvars;
run;

/**************************/
/* Create Macro Variables */
/**************************/

%let target = BAD;
%put &target;

proc sql noprint;
    select variable into :inputs separated by ' '
    from selectedvars;
quit;

%put &inputs;

proc sql noprint;
    select variable into :nominals separated by ' '
    from selectedvars
    where type="CLASS";
quit;

%put &nominals;

proc sql noprint;
    select variable into :intervals separated by ' '
    from selectedvars
    where type="INTERVAL";
quit;

%put &intervals;

/****************/
/* Build Models */
/****************/

proc logistic data=hmeq (where=(_PARTIND_=1));
    class &nominals;
    model &target = &inputs;
    code file="/workspaces/myfolder/mystuff/lr_score_code.sas";
run;

proc treesplit data=hmeq (where=(_PARTIND_=1));
    class &target &nominals;
    model &target = &inputs;
    prune costcomplexity;
    code file="/workspaces/myfolder/mystuff/dt_score_code.sas";
run;

proc forest data=hmeq (where=(_PARTIND_=1)) ntrees=100;
    target &target / level=nominal;
    input &nominals / level=nominal;
    input &intervals / level=interval;
    code file="/workspaces/myfolder/mystuff/rf_score_code.sas";
run;

proc gradboost data=hmeq (where=(_PARTIND_=1)) ntrees=100;
    target &target / level=nominal;
    input &nominals / level=nominal;
    input &intervals / level=interval;
    code file="/workspaces/myfolder/mystuff/gb_score_code.sas";
run;

proc nnet data=hmeq (where=(_PARTIND_=1));
    target &target / level=nominal;
    input &nominals / level=nominal;
    input &intervals / level=interval;
    architecture mlp;
    hidden 25 / act=tanh; 
    hidden 25 / act=tanh;
    optimization algorithm=lbfgs maxiter=100;
    train outmodel=nnout;
    code file="/workspaces/myfolder/mystuff/nn_score_code.sas";
run;

/*************************/
/* Score Validation Data */
/*************************/

data lr_scored;
    set hmeq (where=(_PARTIND_=0));
    %include "/workspaces/myfolder/mystuff/lr_score_code.sas";
run;

data dt_scored;
    set hmeq (where=(_PARTIND_=0));
    %include "/workspaces/myfolder/mystuff/dt_score_code.sas";
run;

data rf_scored;
    set hmeq (where=(_PARTIND_=0));
    %include "/workspaces/myfolder/mystuff/rf_score_code.sas";
run;

data gb_scored;
    set hmeq (where=(_PARTIND_=0));
    %include "/workspaces/myfolder/mystuff/gb_score_code.sas";
run;

data nn_scored;
    set hmeq (where=(_PARTIND_=0));
    %include "/workspaces/myfolder/mystuff/nn_score_code.sas";
run;

title "Variables Created For Each Model";
proc print data=nn_scored (obs=5);
    var BAD I_BAD P_BAD1 P_BAD0;
run;
title;

/*****************/
/* Assess Models */
/*****************/

proc assess data=lr_scored rocout=lr_roc ncuts=100;
    var P_BAD1;
    target &target / event="1" level=nominal;
run;

proc assess data=dt_scored rocout=dt_roc ncuts=100;
    var P_BAD1;
    target &target / event="1" level=nominal;
run;

proc assess data=rf_scored rocout=rf_roc ncuts=100;
    var P_BAD1;
    target &target / event="1" level=nominal;
run;

proc assess data=gb_scored rocout=gb_roc ncuts=100;
    var P_BAD1;
    target &target / event="1" level=nominal;
run;

proc assess data=nn_scored rocout=nn_roc ncuts=100;
    var P_BAD1;
    target &target / event="1" level=nominal;
run;

/***************/
/* ROC Graphic */
/***************/

data assess;
    merge lr_roc(rename=(_Sensitivity_=lr_sensitivity _FPR_=lr_fpr))
          dt_roc(rename=(_Sensitivity_=dt_sensitivity _FPR_=dt_fpr))
          rf_roc(rename=(_Sensitivity_=rf_sensitivity _FPR_=rf_fpr))
          gb_roc(rename=(_Sensitivity_=gb_sensitivity _FPR_=gb_fpr))
          nn_roc(rename=(_Sensitivity_=nn_sensitivity _FPR_=nn_fpr));
    keep _Cutoff_ lr_sensitivity lr_fpr
                  dt_sensitivity dt_fpr
                  rf_sensitivity rf_fpr
                  gb_sensitivity gb_fpr
                  nn_sensitivity nn_fpr;
run;

title "ROC on Validation Partition";
proc sgplot data=assess;
    series x=lr_fpr y=lr_sensitivity / legendlabel="Logistic Regression";
    series x=dt_fpr y=dt_sensitivity / legendlabel="Decision Tree";
    series x=rf_fpr y=rf_sensitivity / legendlabel="Randm Forest";
    series x=gb_fpr y=gb_sensitivity / legendlabel="Gradient Boosting";
    series x=nn_fpr y=nn_sensitivity / legendlabel="Neural Network";
    lineparm x=0 y=0 slope=1 / legendlabel="Baseline";
run;
title;

/************/
/* Accuracy */
/************/

data lr_roc;
    set lr_roc;
    Model = "LR";
run;

data dt_roc;
    set dt_roc;
    Model = "DT";
run;

data rf_roc;
    set rf_roc;
    Model = "RF";
run;

data gb_roc;
    set gb_roc;
    Model = "GB";
run;

data nn_roc;
    set nn_roc;
    Model = "NN";
run;

data acc;
    set lr_roc dt_roc rf_roc gb_roc nn_roc ;
    where round(_Cutoff_, 0.01)=0.5;
    Accuracy = round(_ACC_,0.001);
    AUC = round(_C_,0.001);
    keep Model Accuracy AUC;
run;

proc sort data=acc out=acc;
    by Accuracy;
run;

title "Accuracy and Area Under the Curve";
proc print data=acc;
run;
title;
