classdef SingleNetwork
   properties
      network

      trainResults
      MeanResult
      logLevel = 2
   end
   methods
       function obj = SingleNetwork(hiddenUnits, transferFcn1, transferFcn2,performFcn, trainRatio,valRatio,testRatio,trainFcn,epochs,momentum, learning_rate)
           if nargin >1
                obj.network = feedforwardnet(hiddenUnits);
                            
                obj.network.layers{1}.transferFcn = transferFcn1;
                obj.network.layers{2}.transferFcn = transferFcn2;
               
                obj.network.divideFcn = "dividerand";       % divideFCN allow to change the way the data is divided into training, validation and test data sets. 
                obj.network.divideParam.trainRatio = trainRatio;   % Ratio of data used as training set    0.8；0.4；0.1 
                obj.network.divideParam.valRatio   = valRatio;   % Ratio of data used as validation set  0.1；0.2；0.1
                obj.network.divideParam.testRatio  = testRatio;   % Ratio of data used as test set        0.1；0.4；0.8            
                obj.network.trainFcn = trainFcn;
                obj.network.trainParam.max_fail = 30;   % validation check parameter
                obj.network.trainParam.min_grad = 1e-9; % minimum performance gradient 
                obj.network.trainParam.epochs = epochs;   % number of epochs parameter 
                obj.network.trainParam.mc     = momentum;    % momentum parameter
                obj.network.trainParam.lr     = learning_rate;   % learning rate parameter
               
                obj.network.performFcn= performFcn; % crossentropy, mse
                
                obj.network.name = sprintf('FFNN_%dHU_TFcn2-%s_DR[%d-%d-%d]_PFcn-%s',hiddenUnits,transferFcn2, ...
                    trainRatio*100,valRatio*100,testRatio*100,performFcn);
                obj.trainResults = TrainResult();
            end
       end
       function obj = Train(obj, X,T,times)
            for t = 1:times
                if obj.logLevel >=1
                    fprintf('Start Train %s [%d] Time\n',obj.network.name,t);
                end
                tic
                obj.network = configure(obj.network,X,T);
                [obj.network,tr,Y,E] = train(obj.network,X,T);
                toc
                obj.trainResults(t) = TrainResult(T,tr,Y,toc-tic);
                obj.trainResults(end).ShowResult();
            end
       end
   end
end