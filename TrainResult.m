classdef TrainResult
    properties
        TR
        Train_Y
        Val_Y
        Test_Y
        Train_T
        Val_T
        Test_T
        
        Class_Train_Y
        Class_Val_Y
        Class_Test_Y
        Class_Train_T
        Class_Val_T
        Class_Test_T
    
        Matches_Train
        Accuracy_Train
    
        Matches_Val
        Accuracy_Val
    
        Matches_Test
        Accuracy_Test
    
        Perform_Train
        Perform_Val
        Perform_Test
        
        TrainTime
    end
    methods
        function obj = TrainResult(T,TR,Y,TT)
            % Store training result of a network
            % T is the True output value
            % TR is the training results
            % Y is the predict output
            % TT is the training time
            if nargin >0
                obj.TR=TR;
        
                obj.Train_Y = Y(:, TR.trainInd); 
                obj.Val_Y   = Y(:, TR.valInd);
                obj.Test_Y  = Y(:, TR.testInd);
        
                obj.Train_T = T(:, TR.trainInd); 
                obj.Val_T   = T(:, TR.valInd);
                obj.Test_T  = T(:, TR.testInd);
        
                [~,obj.Class_Train_Y] = max(obj.Train_Y, [], 1);
                [~,obj.Class_Val_Y]   = max(obj.Val_Y, [], 1);
                [~,obj.Class_Test_Y]  = max(obj.Test_Y, [], 1);
        
                [~,obj.Class_Train_T] = max(obj.Train_T, [], 1);
                [~,obj.Class_Val_T]   = max(obj.Val_T, [], 1);
                [~,obj.Class_Test_T]  = max(obj.Test_T, [], 1);
                
    
                obj.Matches_Train  = sum((obj.Class_Train_Y ==obj.Class_Train_T));
                obj.Accuracy_Train = obj.Matches_Train / size(obj.Class_Train_T,2);
        
                obj.Matches_Val  = sum((obj.Class_Val_Y ==obj.Class_Val_T));
                obj.Accuracy_Val = obj.Matches_Val / size(obj.Class_Val_T,2);
        
                obj.Matches_Test  = sum((obj.Class_Test_Y ==obj.Class_Test_T));
                obj.Accuracy_Test = obj.Matches_Test / size(obj.Class_Test_T,2);
        
                obj.Perform_Train = mean(TR.perf);
                obj.Perform_Val   = mean(TR.vperf);
                obj.Perform_Test  = mean(TR.tperf);
    
                obj.TrainTime = TT;
            end
    
        end
        function obj = Save(obj,path,filename)
            Train_Y      = obj.Train_Y;
            Val_Y        = obj.Val_Y;
            Test_Y       = obj.Test_Y;
            Train_T      = obj.Train_T;
            Val_T        = obj.Val_T;
            Test_T       = obj.Test_T;

            ClassTable_Train = table(obj.TR.trainInd',obj.Class_Train_T',obj.Class_Train_Y','VariableNames',{'Index','Target','Predict'});
            ClassTable_Val   = table(obj.TR.valInd',obj.Class_Val_T',obj.Class_Val_Y','VariableNames',{'Index','Target','Predict'});
            ClassTable_Test  = table(obj.TR.testInd',obj.Class_Test_T',obj.Class_Test_Y','VariableNames',{'Index','Target','Predict'});

            Matches     = [obj.Matches_Train;obj.Matches_Val;obj.Matches_Test];
            Accuracy    = [obj.Accuracy_Train;obj.Accuracy_Val;obj.Accuracy_Test];
            Performance = [obj.Perform_Train;obj.Perform_Val;obj.Perform_Test];
            ResultTable = table(Matches,Accuracy,Performance,'VariableNames',{'Matches','Accuracy','Performance'},'RowNames',{'Train','Val','Test'});
            TrainTime   = obj.TrainTime;
            f = sprintf('%s/%s',path,filename);
            if ~exist(path, 'dir')
                mkdir(path);
            end
            save(f,'Train_Y', 'Val_Y','Test_Y', ...
                'Train_T','Val_T','Test_T', ...
                'ClassTable_Train','ClassTable_Val','ClassTable_Test','ResultTable', 'TrainTime');
            
        end
        function obj = ShowResult(obj)
            fprintf('Train Result:\tCount= %d\tMatches= %d\tAccuracy= %.3f%%\tPerformance= %.3f%%\n',length(obj.TR.trainInd),obj.Matches_Train,obj.Accuracy_Train*100,obj.Perform_Train*100);
            fprintf('Val   Result:\tCount= %d\tMatches= %d\tAccuracy= %.3f%%\tPerformance= %.3f%%\n',length(obj.TR.valInd)  ,obj.Matches_Val  ,obj.Accuracy_Val*100  ,obj.Perform_Val*100);
            fprintf('Test  Result:\tCount= %d\tMatches= %d\tAccuracy= %.3f%%\tPerformance= %.3f%%\n',length(obj.TR.testInd) ,obj.Matches_Test ,obj.Accuracy_Test*100 ,obj.Perform_Test*100);
            fprintf('====================Train Time: %.3fs====================\n',obj.TrainTime);
        end
    end
end