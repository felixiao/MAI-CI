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
       function obj = CustomNetwork(inputs,outputs)
        if(nargin == 2)
%             obj.Inputs = inputs;
%             obj.Outputs = outputs;
            obj.Net = network( 1, 2, ...          % Number of Inputs, Layers
                [1; 0], ...     % Bias Connect
                [1; 0], ...     % Input Connect
                [0 0; 1 0], ... % Layer Connect
                [0 1]);         % Output Connect
            obj.Net.layers{1}.size = 50;            % hidden layer size = 50, 200, 500
            obj.Net.layers{1}.transferFcn = 'logsig'; % hidden layer logsig, tansig
            obj.Net.layers{2}.transferFcn = 'logsig'; % output layer logsig, softmax
            
            obj.Net.divideFcn = 'dividerand';       % divideFCN allow to change the way the data is 
                                                    % divided into training, validation and test data sets. 
            % 80, 10, 10; 40, 20, 40; 10, 10, 80
            obj.Net.divideParam.trainRatio = 0.8;   % Ratio of data used as training set
            obj.Net.divideParam.valRatio = 0.1;     % Ratio of data used as validation set
            obj.Net.divideParam.testRatio = 0.1;    % Ratio of data used as test set

%             obj.Net.trainParam.max_fail = 6;    % validation check parameter
%             obj.Net.trainParam.epochs=2000;     % number of epochs parameter 
%             obj.Net.trainParam.min_grad = 1e-5; % minimum performance gradient 

            obj.Net.performFcn='crossentropy'; % crossentropy, mse
            
            obj.Net.trainFcn='trainlm';     % Levenberg-Marquardt
            %obj.Net.trainFcn='traingdm';    % Gradient Descent with momentum
            %obj.Net.trainFcn='traingdx';    % Gradient descent with momentum and adaptive
                
            obj.Net.trainParam.mc = 0.8;    % momentum parameter
            obj.Net.trainParam.lr = 0.01;   % learning rate parameter

            %obj.Net= configure(obj.Net,obj.Inputs,obj.Outputs);
            %view(obj.Net)
        end
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