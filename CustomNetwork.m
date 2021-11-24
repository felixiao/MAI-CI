classdef CustomNetwork
   properties
      % Hyper Parameters
      % Practice
      Functions    = ["logsig", "logsig", "mse"; "logsig", "softmax", "crossentropy"]; % default logsig logsig mse
      Num_HiddenLayerUnits = [50,200,500];% default 50
      Divide_Ratios        = [0.8,0.1,0.1; 0.4,0.2,0.4; 0.1,0.1,0.8]; %default 0.8
      TrainTimes           = 3; % default 3
      

      % Tune
      LearningRates  = [0.01, 0.1, 0.001, 0.0001];        % default 0.01
      Momentums      = [ 0.8, 0.6,   0.7,    0.9,  0.99]; % default 0.8
      Num_Epochs     = [1000, 5000,  1500,  2000];         % default 1000
      TrainFunctions = ["traingdm", "traingdx","trainscg"]; % default traingdm

      % Datas
      Inputs
      Outputs

      % Results

      %Net
      Net
      Networks 
      NetworkType

      logLevel = 1
   end
   methods
       function obj = CustomNetwork(X,T)
           % Custom Network (feedforward)
           obj.Inputs  = X;
           obj.Outputs = T;
           obj.NetworkType = 'FFNN';
           obj.Networks = SingleNetwork();
           obj = obj.Setup();
           fprintf('Network Count: %d\n',length(obj.Networks));
       end
       function obj = Setup(obj)
           netindex = 0;
           for transferFcnIdx = 1:obj.Functions.size(1)
                for numHiddenLayer = obj.Num_HiddenLayerUnits
                    for divideRatio = 1:length(obj.Divide_Ratios)
                        netindex = netindex+1;
                        if obj.logLevel >= 2
                            fprintf('\nSetup Network: [%d]========================> \nHyperParameters:\n\tLayer [1] TransferFcn=[%s]\n\tLayer [2] TransferFcn=[%s]\n\tCost Function [%s]\n\t[%d] Hidden Units\n\tDivide Into [TrainSet=%d, ValidateSet=%d, TestSet=%d]\n' ...
                                ,netindex,obj.Functions(transferFcnIdx,1),obj.Functions(transferFcnIdx,2),obj.Functions(transferFcnIdx,3), ...
                                numHiddenLayer,obj.Divide_Ratios(divideRatio,1)*100,obj.Divide_Ratios(divideRatio,2)*100,obj.Divide_Ratios(divideRatio,3)*100);
                        end
                        net = SingleNetwork(obj.NetworkType,numHiddenLayer,obj.Functions(transferFcnIdx,1),obj.Functions(transferFcnIdx,2),obj.Functions(transferFcnIdx,3),obj.Divide_Ratios(divideRatio,1),obj.Divide_Ratios(divideRatio,2),obj.Divide_Ratios(divideRatio,3), ...
                            'traingdm',50,0.8,0.01);
                        
                        obj.Networks(netindex) = net;
            
                        if obj.logLevel >=1
                            fprintf('[%d]\tName: %s\n',netindex,obj.Networks(end).network.name);
                        end
                    end
                end
           end
       end
       function obj = TrainAll(obj)
           for index = 1:length(obj.Networks)
                obj = obj.Train(index);
           end
       end
       function obj = Train(obj,index)
           fprintf('Network Info: %s',obj.Networks(index).network.userdata.note); 
           obj.Networks(index) = obj.Networks(index).Train(obj.Inputs,obj.Outputs,obj.TrainTimes);
       end
       
   end
end