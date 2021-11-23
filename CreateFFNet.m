function r = CreateFFNet(I,O)
    r = feedforwardnet(50,'traingdm');
    r = configure(r,I,O);
    view(r);
%     r.name                   = 'Test';
%     r.layers{1}.size         = 10;
%     r.layers{1}.transferFcn  = 'logsig';
%     r.layers{2}.transferFcn  = 'logsig';
%     r.divideFcn              = 'dividerand';
%     r.divideParam.trainRatio = 0.8;
%     r.divideParam.valRatio   = 0.1;
%     r.divideParam.testRatio  = 0.1;
%     r.trainFcn               = 'trainlm';
%     r.trainParam.max_fail    = 6;       % validation check parameter
%     r.trainParam.epochs      = 2000;    % number of epochs parameter 
%     r.trainParam.min_grad    = 1e-5;    % minimum performance gradient 
%     r.trainParam.mc          = 0.8;     % momentum parameter
%     r.trainParam.lr          = 0.01;    % learning rate parameter
%     r.performFcn             = 'crossentropy'; % crossentropy, mse    
%     r = configure(r,I,O);
%     view(r);
end