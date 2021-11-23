function r = CreatePNet(I,O)
    r = patternnet(50);
    r.layers{1}.transferFcn = 'logsig';
    r.layers{2}.transferFcn = 'softmax';
    r.adaptFcn = 'adaptwb';
    r.divideFcn = 'dividerand';       % divideFCN allow to change the way the data is divided into training, validation and test data sets. 
    r.divideParam.trainRatio = 0.8;   % Ratio of data used as training set    0.8；0.4；0.1 
    r.divideParam.valRatio = 0.1;     % Ratio of data used as validation set  0.1；0.2；0.1
    r.divideParam.testRatio = 0.1;    % Ratio of data used as test set        0.1；0.4；0.8 
    r.trainFcn='traingdm';
    r.trainParam.epochs = 500;
    r.trainParam.min_grad = 1e-5;
    r.trainParam.max_fail = 6;
    r.trainParam.mc = 0.8;
    r.trainParam.lr = 0.01;
    r.performFcn= 'crossentropy';
    r = configure(r,I,O);
    view(r);
end