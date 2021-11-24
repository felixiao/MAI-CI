classdef SingleNetwork
   properties
      network

      trainResults
      MeanResult
      logLevel = 2
      networkType
   end
   methods
       function obj = SingleNetwork(Type,hiddenUnits, transferFcn1, transferFcn2,performFcn, trainRatio,valRatio,testRatio,trainFcn,epochs,momentum, learning_rate)
           if nargin >1
                obj.networkType = Type;
                if obj.networkType == "FFNN"
                    obj.network = feedforwardnet(hiddenUnits);
                elseif obj.networkType == "PTNN"
                    obj.network = patternnet(hiddenUnits);
                end
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
                
                obj.network.name = sprintf('%s_%dHU_TFcn2-%s_DR[%d-%d-%d]_PFcn-%s',obj.networkType,hiddenUnits,transferFcn2, ...
                    trainRatio*100,valRatio*100,testRatio*100,performFcn);
                obj.network.userdata.note = sprintf(['\nFeedForward Network\n' ...
                    '\tHidden Units \t= %d\n\tTransferFcn1 \t= %s\n\tTransferFcn2 \t= %s\n' ...
                    '\tDivideRatio \t= [%d-%d-%d]\n\tPerformFcn \t= %s\n\tTrainFcn \t= %s\n' ...
                    '\tEpochs \t\t= %d\n\tLearningRate \t= %.3f\n\tMomentum \t= %.3f\n'],hiddenUnits,transferFcn1,transferFcn2, ...
                     trainRatio*100,valRatio*100,testRatio*100,performFcn,trainFcn,epochs,learning_rate,momentum);
                obj.trainResults = TrainResult();
                
            end
       end
       function obj = Train(obj, X,T,times)
            obj.network = configure(obj.network,X,T);
            for t = 1:times
                obj.network = init(obj.network);
                if obj.logLevel >=1
                    fprintf('\nStart Train %s [%d] Time\n',obj.network.name,t);
                end
                tic
                [obj.network,tr,Y,E] = train(obj.network,X,T);
                obj.trainResults(t) = TrainResult(T,tr,Y,toc);
                obj.trainResults(t).ShowResult();
            end
       end
   end
end