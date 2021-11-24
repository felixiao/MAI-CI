classdef SingleNetwork
   properties
       networkType
       network
       trainResults
       MeanResult
       logLevel = 2
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
                obj.network.trainParam.max_fail = 500;   % validation check parameter
                obj.network.trainParam.min_grad = 1e-10; % minimum performance gradient 
                obj.network.trainParam.epochs = epochs;   % number of epochs parameter 
                obj.network.trainParam.mc     = momentum;    % momentum parameter
                obj.network.trainParam.lr     = learning_rate;   % learning rate parameter
                
                obj.network.trainParam.goal   = 1e-3;
                
                obj.network.plotFcns{1} = 'plotperform';
                obj.network.plotFcns{2} = 'ploterrhist';
                obj.network.plotFcns{3} = 'plotconfusion';
                obj.network.plotFcns{4} = 'plotroc';
                obj.network.trainParam.show= 50;
%                 obj.network.trainParam.showCommandLine = true;


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
            if obj.logLevel >=1
                fprintf('\nStart Train: %s \n', obj.network.name);
            end
            obj.network = configure(obj.network,X,T);
            for t = 1:times
                obj.network = init(obj.network);
                tic
                [obj.network,tr,Y,E] = train(obj.network,X,T);
                obj.trainResults(t) = TrainResult(T,tr,Y,toc);
                obj.trainResults(t)
                if obj.logLevel >=1
                    fprintf('[%d] Iteration\t',t);
                    obj.trainResults(t).ShowResult();
                end
%                 path = sprintf('./Results/%s',obj.network.name);
%                 obj.trainResults(t).Save(path);
            end
       end
       
   end
end