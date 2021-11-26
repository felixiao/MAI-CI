classdef SingleNetwork
   properties
       ID
       Tag
       Type
       network
       trainResults
       ResultTable
       logLevel = 2
       Path
       TotalTrainTime
   end
   methods
       function obj = SingleNetwork(Type,hiddenUnits, transferFcn1, transferFcn2,performFcn, trainRatio,valRatio,testRatio,trainFcn,epochs,momentum, learning_rate, max_fail)
           if nargin >1
                obj.Type = Type;
                if obj.Type == "FFNN"
                    typename = 'FeedForward Neural Network';
                    obj.network = feedforwardnet(hiddenUnits);
                elseif obj.Type == "PTNN"
                    typename = 'Pattern Neural Network';
                    obj.network = patternnet(hiddenUnits);
                end
                obj.network.layers{1}.transferFcn = transferFcn1;
                obj.network.layers{2}.transferFcn = transferFcn2;
               
                obj.network.divideFcn = "dividerand";       % divideFCN allow to change the way the data is divided into training, validation and test data sets. 
                obj.network.divideParam.trainRatio = trainRatio;   % Ratio of data used as training set    0.8；0.4；0.1 
                obj.network.divideParam.valRatio   = valRatio;   % Ratio of data used as validation set  0.1；0.2；0.1
                obj.network.divideParam.testRatio  = testRatio;   % Ratio of data used as test set        0.1；0.4；0.8            
                obj.network.trainFcn = trainFcn;
                obj.network.trainParam.max_fail = max_fail;   % validation check parameter
                obj.network.trainParam.min_grad = 1e-8; % minimum performance gradient 
                obj.network.trainParam.epochs = epochs;   % number of epochs parameter 
                obj.network.trainParam.mc     = momentum;    % momentum parameter
                obj.network.trainParam.lr     = learning_rate;   % learning rate parameter
              
                obj.network.plotFcns{1} = 'plotperform';
                obj.network.plotFcns{2} = 'ploterrhist';
                obj.network.plotFcns{3} = 'plotregression';
                obj.network.plotFcns{4} = 'plotroc';
%                 obj.network.trainParam.show= 50;
                 obj.network.trainParam.showWindow= false;
%                 obj.network.trainParam.showCommandLine = true;

                obj.network.performFcn= performFcn; % crossentropy, mse
                obj.network.name = sprintf('%s-%d-%s-%s-[%d-%d-%d]-%.4f-%.3f-%d-%s-%d',obj.Type,hiddenUnits,transferFcn2,performFcn, ...
                    trainRatio*100,valRatio*100,testRatio*100,learning_rate,momentum,epochs,trainFcn,max_fail);
                obj.network.userdata.note = sprintf(['\n%s\n' ...
                    '\tHidden Units \t= %d\n\tTransferFcn1 \t= %s\n\tTransferFcn2 \t= %s\n' ...
                    '\tDivideRatio \t= [%d-%d-%d]\n\tPerformFcn \t= %s\n\tTrainFcn \t= %s\n' ...
                    '\tEpochs \t\t= %d\n\tLearningRate \t= %.3f\n\tMomentum \t= %.3f\n\tMaxFail\t\t= %d\n'],typename,hiddenUnits,transferFcn1,transferFcn2, ...
                     trainRatio*100,valRatio*100,testRatio*100,performFcn,trainFcn,epochs,learning_rate,momentum,max_fail);
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
                if obj.logLevel >=2
                    fprintf('====================[%d] Iteration====================\n',t);
                    obj.trainResults(t).ShowResult();

                end
                filename = sprintf('%d.mat',t);
                obj.trainResults(t).Save(obj.Path,filename);
            end
            
            if obj.logLevel >=1
                obj = ShowResults(obj);
                fprintf('====================End Train Time: %.3f========================\n\n',obj.TotalTrainTime);
            end
            
       end
       function obj = ShowResults(obj)
            matchTrain = floor(mean([obj.trainResults.Matches_Train]));
            matchVal   = floor(mean([obj.trainResults.Matches_Val]));
            matchTest  = floor(mean([obj.trainResults.Matches_Test]));

            accTrain = mean([obj.trainResults.Accuracy_Train]);
            accVal   = mean([obj.trainResults.Accuracy_Val]);
            accTest  = mean([obj.trainResults.Accuracy_Test]);

            perfTrain = mean([obj.trainResults.Perform_Train]);
            perfVal   = mean([obj.trainResults.Perform_Val]);
            perfTest  = mean([obj.trainResults.Perform_Test]);

            Matches     = [matchTrain;matchVal;matchTest];
            Accuracy    = [accTrain;accVal;accTest];
            Performance = [perfTrain;perfVal;perfTest];
            Table       = table(Matches,Accuracy,Performance,'RowNames',{'Train','Val','Test'});
            obj.ResultTable = Table;

            TotalTTime = sum([obj.trainResults.TrainTime]);
            obj.TotalTrainTime = TotalTTime;
            f = sprintf('%s/result.mat',obj.Path);
            info = sprintf('%s/Network_Configure_Info.txt',obj.Path);
            txt = fopen(info,'w');
            fprintf(txt,'%s\n',obj.network.userdata.note);
            save(f,'Table','TotalTTime');
            
            t = tiledlayout(3,2,'TileSpacing','tight','Padding','tight');
            
            nexttile;
            plot(1:length(obj.trainResults),[obj.trainResults.Perform_Train],'r-o', ...
                1:length(obj.trainResults),[obj.trainResults.Perform_Val],'g-+', ...
                1:length(obj.trainResults),[obj.trainResults.Perform_Test],'b-*');
            xlabel('Iteration');
            ylabel('Performance');
            title('Performance at each iteration');
            legend({'Train','Validation','Test'},'Location','best');

            nexttile;
            plot(1:length(obj.trainResults),[obj.trainResults.Accuracy_Train],'r-o', ...
                1:length(obj.trainResults),[obj.trainResults.Accuracy_Val],'g-+', ...
                1:length(obj.trainResults),[obj.trainResults.Accuracy_Test],'b-*');
            xlabel('Iteration');
            ylabel('Accuracy');
            title(' Accuracy at each iteration');
            legend({'Train','Validation','Test'},'Location','best');

            nexttile(3,[2 2]);

            confusion = confusionchart(obj.trainResults(1).Class_Test_T,obj.trainResults(1).Class_Test_Y);
            title('Confusion Matrix');

            title(t,obj.network.name)
            saveas(gca,sprintf('%s/Results.jpg',obj.Path));
            close all;

            fprintf('====================Train Result====================\n');
            fprintf('Train Mean Result:\tMatches=%d\tAccuracy= %.3f%%\tPerformance= %.4f%%\n',matchTrain,accTrain*100,perfTrain*100);
            fprintf('Val   Mean Result:\tMatches=%d\tAccuracy= %.3f%%\tPerformance= %.4f%%\n',matchVal,accVal*100,perfVal*100);
            fprintf('Test  Mean Result:\tMatches=%d\tAccuracy= %.3f%%\tPerformance= %.4f%%\n',matchTest,accTest*100,perfTest*100);
       end
   end
end