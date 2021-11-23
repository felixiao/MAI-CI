classdef CustomNetwork
   properties
      %Hyper Parameters
      Value {mustBeNumeric}
      %Datas
      Inputs
      Outputs
      TrainData
      %Net
      Net
   end
   methods
       function obj = CustomNetwork(X,T)
           % Custom Network (feedforward)
           % X is the input
           % T is the target
           obj.Net = network( 1, 2,[1; 0],[1; 0],[0 0; 1 0],[0 1]);
           obj.Net.name = 'Test';
           obj.Net.layers{1}.size = 50;            % hidden layer size = 50, 200, 500
           obj.Net.layers{1}.transferFcn = "logsig"; % hidden layer logsig, tansig
           obj.Net.layers{2}.transferFcn = "logsig"; % output layer logsig, softmax
           obj.Net.divideFcn = "dividerand";       % divideFCN allow to change the way the data is divided into training, validation and test data sets. 
           obj.Net.divideParam.trainRatio = 0.8;   % Ratio of data used as training set    0.8；0.4；0.1 
           obj.Net.divideParam.valRatio = 0.1;     % Ratio of data used as validation set  0.1；0.2；0.1
           obj.Net.divideParam.testRatio = 0.1;    % Ratio of data used as test set        0.1；0.4；0.8            
           obj.Net.trainFcn="trainlm";         % Levenberg-Marquardt traingdm; traingdx
           obj.Net.trainParam.max_fail = 6;    % validation check parameter
           obj.Net.trainParam.epochs=2000;     % number of epochs parameter 
           obj.Net.trainParam.min_grad = 1e-5; % minimum performance gradient 
           obj.Net.trainParam.mc = 0.8;    % momentum parameter
           obj.Net.trainParam.lr = 0.01;   % learning rate parameter
           obj.Net.performFcn= "crossentropy"; % crossentropy, mse
           obj.Net = configure(obj.Net,X,T);
           obj.Inputs = X;
           obj.Outputs = T;
       end
 
       function obj = Train(obj)
           [obj.Net,tr,Y,E]  = train(obj.Net,obj.Inputs,obj.Outputs);
           fprintf('Accuracy: %f\n',100-100*sum(abs((Y>0.5)-T))/length(T))
       end
       function r = roundOff(obj)
           r = round([obj.Value],2);
       end
       function r = multiplyBy(obj,n)
           r = [obj.Value] * n;
       end
   end
end