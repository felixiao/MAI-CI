classdef CustomNetwork
   properties
      % Hyper Parameters
      % Practice
      Functions    = ["logsig", "logsig", "mse"; "logsig", "softmax", "crossentropy"]; % default logsig logsig mse
      Num_HiddenLayerUnits = [50,200,500];% default 50
      Divide_Ratios        = [0.4,0.2,0.4; 0.8,0.1,0.1; 0.1,0.1,0.8]; %default 0.4 0.2 0.4
      TrainTimes           = 3; % default 3
      
      % Pre Train
      LearningRates  = [0.001, 0.01, 0.1];        % default 0.01
      Momentums      = [ 0.6, 0.8,  0.99]; % default 0.8
      Num_Epochs     = [500, 1000,  2000];         % default 1000
      TrainFunctions = ["trainscg"]; % default trainscg
      Max_Fails      = [50, 200, 500]; % default 500
      PreTrainNetworks
      PreTrainParameters
      PreTSameParam
      PreTDiffParam
      PreTParamNames
      TotalPreTrainTime

      % Datas
      Inputs;
      Outputs;

      %Net
      Networks
      NetworkType
      Parameters
      SameParam
      DiffParam
      ParamNames
      TotalTrainTime
      logLevel = 1
   end
   methods
       function obj = CustomNetwork(X,T)
           % Custom Network (feedforward)
           obj.Inputs  = X;
           obj.Outputs = T;
           obj.NetworkType = 'FFNN';
           obj.Networks = SingleNetwork();
           obj.PreTrainNetworks = SingleNetwork();
            % Pre training
           obj = obj.PreTrainSetup();
           fprintf('===================Pretrain Network Count: %d=================\n',length(obj.PreTrainNetworks));
            % Choose best configuration
           obj = obj.Setup();
           fprintf('===================Network Count: %d==========================\n',length(obj.Networks));
       end
       
       function obj = PreTrainSetup(obj)
           obj.PreTSameParam = struct('LR001',[],'LR01',[],'LR0001',[], ...
                'MC8',[],'MC6',[],'MC99',[], ...
                'EP1000',[],'EP500',[],'EP2000',[]);
           obj.PreTDiffParam = struct('LR',[],'MC',[],'EP',[]);
           obj.PreTParamNames = struct('LR',[],'MC',[],'EP',[]);
           obj.PreTParamNames.LR = ["MC6-EP500";"MC6-EP1000";"MC6-EP2000";"MC8-EP500";"MC8-EP1000";"MC8-EP2000";"MC99-EP500" ;"MC99-EP1000" ; "MC99-EP2000"];
           obj.PreTParamNames.MC = ["LR0001-EP500";"LR0001-EP1000";"LR0001-EP2000";"LR001-EP500";"LR001-EP1000";"LR001-EP2000";"LR01-EP500";"LR01-EP1000";"LR01-EP2000"];
           obj.PreTParamNames.EP = ["LR0001-MC6";"LR0001-MC8";"LR0001-MC99";"LR001-MC6";"LR001-MC8";"LR001-MC99";"LR01-MC6";"LR01-MC8";"LR01-MC99"];
           netindex = 0;
           for trainFcn = obj.TrainFunctions
               for lr = obj.LearningRates
                    for mc = obj.Momentums
                        for ep = obj.Num_Epochs
    
                            netindex = netindex+1;
                            if obj.logLevel >=1
                                fprintf('\nSetup Pretrain Network:========================> \n');
                            end
                            if obj.logLevel >= 2
                                fprintf('HyperParameters:\n\tLayer [1] TransferFcn=[%s]\n\tLayer [2] TransferFcn=[%s]\n\tCost Function [%s]\n\t[%d] Hidden Units\n\tDivide Into [TrainSet=%d, ValidateSet=%d, TestSet=%d]\n' ...
                                    ,obj.Functions(transferFcnIdx,1),obj.Functions(transferFcnIdx,2),obj.Functions(transferFcnIdx,3), ...
                                    numHiddenLayer,obj.Divide_Ratios(divideRatio,1)*100,obj.Divide_Ratios(divideRatio,2)*100,obj.Divide_Ratios(divideRatio,3)*100);
                            end
                            net = SingleNetwork(obj.NetworkType,50,'logsig','softmax','crossentropy',0.4,0.2,0.4, ...
                                trainFcn,ep,mc,lr,500);
                            
                            desiredFolder = './PreTrainResults/';
                            path = strcat(desiredFolder,net.network.name);
                            if ~exist(path, 'dir')
                                mkdir(path);
                            end
                            net.Path = path;
                            net.ID   = netindex;
                            net.Tag  = 'PreTrain';
                            obj.PreTrainNetworks(netindex) = net;
                            
                            if lr == 0.01
                                obj.PreTSameParam.LR001(end+1) = netindex;
                            elseif lr == 0.1
                                obj.PreTSameParam.LR01(end+1) = netindex;
                            elseif lr == 0.001
                                obj.PreTSameParam.LR0001(end+1) = netindex;
                            end
    
                            if mc == 0.8
                                obj.PreTSameParam.MC8(end+1) = netindex;
                            elseif mc == 0.6
                                obj.PreTSameParam.MC6(end+1) = netindex;
                            elseif mc == 0.99
                                obj.PreTSameParam.MC99(end+1) = netindex;
                            end
    
                            if ep == 1000
                                obj.PreTSameParam.EP1000(end+1) = netindex;
                            elseif ep == 500
                                obj.PreTSameParam.EP500(end+1) = netindex;
                            elseif ep == 2000
                                obj.PreTSameParam.EP2000(end+1) = netindex;
                            end
    
                            if obj.logLevel >=1
                                fprintf('[%d]\tName: %s\n',netindex,obj.PreTrainNetworks(end).network.name);
                            end
                            
                        end
                    end     
               end
           end
           obj = obj.GroupPreTParam();
       end
       function obj = PreTrainAllWith(obj,Param,value)
            networkInd = [];
            for i = 1:length(obj.PreTrainNetworks)
                if Param == "LR" & value == obj.PreTrainNetworks(i).network.trainParam.lr
                    networkInd(end+1) = i;
                elseif Param == "MC" & value == obj.PreTrainNetworks(i).network.trainParam.mc
                    networkInd(end+1) = i;
                elseif Param == "EP" & value ==  obj.PreTrainNetworks(i).network.trainParam.epochs
                    networkInd(end+1) = i;
                elseif Param == "TF" & value == obj.PreTrainNetworks(i).network.trainFcn
                    networkInd(end+1) = i;
                end
            end
            for index = 1:length(networkInd)
               if obj.logLevel >=1
                  fprintf('[%d/%d] ',index,length(networkInd));
               end
               obj = obj.PreTrain(networkInd(index));
           end
           obj.TotalPreTrainTime = sum([obj.PreTrainNetworks(netwokInd).TotalTrainTime]);
           if obj.logLevel >=1
                fprintf('All PreTrain Finished! Time: %.3f\n',obj.TotalPreTrainTime);
           end
       end
       function obj = PreTrain(obj,index)
            if obj.logLevel >=1
                fprintf('Pretrain Network Info:  %s',index,length(obj.PreTrainNetworks),obj.PreTrainNetworks(index).network.userdata.note); 
            end
            obj.PreTrainNetworks(index) = obj.PreTrainNetworks(index).Train(obj.Inputs,obj.Outputs,obj.TrainTimes);
       end
       function obj = PreTrainAll(obj)
            for index = 1:length(obj.PreTrainNetworks)
                if obj.logLevel >=1
                    fprintf('[%d/%d] ',index,length(obj.PreTrainNetworks),obj.PreTrainNetworks(index).network.userdata.note); 
                end
                obj= obj.PreTrain(index);
            end
            obj.TotalPreTrainTime = sum([obj.PreTrainNetworks.TotalTrainTime]);
            if obj.logLevel >=1
                fprintf('All PreTrain Finished! Time: %.3f\n',obj.TotalPreTrainTime);
           end
       end
       function obj = GroupPreTParam(obj)
            LR8_1000 = [];
            LR8_500 = [];
            LR8_2000 = [];
            LR6_1000 = [];
            LR6_500 = [];
            LR6_2000 = [];
            LR99_1000 = [];
            LR99_500 = [];
            LR99_2000 = [];
            
            MC001_1000 = [];
            MC001_500 = [];
            MC001_2000 = [];
            MC01_1000 = [];
            MC01_500  = [];
            MC01_2000 = [];
            MC0001_1000 = [];
            MC0001_500 = [];
            MC0001_2000 = [];

            EP001_8 = [];
            EP001_6 = [];
            EP001_99 = [];
            EP01_8 = [];
            EP01_6 = [];
            EP01_99 = [];
            EP0001_8 = [];
            EP0001_6 = [];
            EP0001_99 = [];
            
            for index = 1:length(obj.PreTrainNetworks)
                if any(obj.PreTSameParam.LR001(:) == index)
                    if any(obj.PreTSameParam.MC8(:) == index)
                        EP001_8(end+1) = index;
                    elseif any(obj.PreTSameParam.MC6(:) == index)
                        EP001_6(end+1) = index;
                    elseif any(obj.PreTSameParam.MC99(:) == index)
                        EP001_99(end+1) = index;
                    end
                    if any(obj.PreTSameParam.EP1000(:) == index)
                        MC001_1000(end+1) = index;
                    elseif any(obj.PreTSameParam.EP500(:) == index)
                        MC001_500(end+1) = index;
                    elseif any(obj.PreTSameParam.EP2000(:) == index)
                        MC001_2000(end+1) = index;
                    end
                elseif any(obj.PreTSameParam.LR01(:) == index)
                    if any(obj.PreTSameParam.MC8(:) == index)
                        EP01_8(end+1) = index;
                    elseif any(obj.PreTSameParam.MC6(:) == index)
                        EP01_6(end+1) = index;
                    elseif any(obj.PreTSameParam.MC99(:) == index)
                        EP01_99(end+1) = index;
                    end
                    if any(obj.PreTSameParam.EP1000(:) == index)
                        MC01_1000(end+1) = index;
                    elseif any(obj.PreTSameParam.EP500(:) == index)
                        MC01_500(end+1) = index;
                    elseif any(obj.PreTSameParam.EP2000(:) == index)
                        MC01_2000(end+1) = index;
                    end
                elseif any(obj.PreTSameParam.LR0001(:) == index)
                    if any(obj.PreTSameParam.MC8(:) == index)
                        EP0001_8(end+1) = index;
                    elseif any(obj.PreTSameParam.MC6(:) == index)
                        EP0001_6(end+1) = index;
                    elseif any(obj.PreTSameParam.MC99(:) == index)
                        EP0001_99(end+1) = index;
                    end
                    if any(obj.PreTSameParam.EP1000(:) == index)
                        MC0001_1000(end+1) = index;
                    elseif any(obj.PreTSameParam.EP500(:) == index)
                        MC0001_500(end+1) = index;
                    elseif any(obj.PreTSameParam.EP2000(:) == index)
                        MC0001_2000(end+1) = index;
                    end
                end
                if any(obj.PreTSameParam.MC8(:) == index)
                    if any(obj.PreTSameParam.EP1000(:) == index)
                        LR8_1000(end+1) = index;
                    elseif any(obj.PreTSameParam.EP500(:) == index)
                        LR8_500(end+1) = index;
                    elseif any(obj.PreTSameParam.EP2000(:) == index)
                        LR8_2000(end+1) = index;
                    end
                elseif any(obj.PreTSameParam.MC6(:) == index)
                    if any(obj.PreTSameParam.EP1000(:) == index)
                        LR6_1000(end+1) = index;
                    elseif any(obj.PreTSameParam.EP500(:) == index)
                        LR6_500(end+1) = index;
                    elseif any(obj.PreTSameParam.EP2000(:) == index)
                        LR6_2000(end+1) = index;
                    end
                elseif any(obj.PreTSameParam.MC99(:) == index)
                    if any(obj.PreTSameParam.EP1000(:) == index)
                        LR99_1000(end+1) = index;
                    elseif any(obj.PreTSameParam.EP500(:) == index)
                        LR99_500(end+1) = index;
                    elseif any(obj.PreTSameParam.EP2000(:) == index)
                        LR99_2000(end+1) = index;
                    end
                end

            end
            obj.PreTDiffParam.LR = [LR6_500;LR6_1000;LR6_2000;LR8_500;LR8_1000;LR8_2000;LR99_500 ;LR99_1000 ; LR99_2000];
            obj.PreTDiffParam.MC = [MC0001_500;MC0001_1000;MC0001_2000;MC001_500;MC001_1000;MC001_2000;MC01_500;MC01_1000;MC01_2000];
            obj.PreTDiffParam.EP = [EP0001_6;EP0001_8;EP0001_99;EP001_6;EP001_8;EP001_99;EP01_6;EP01_8;EP01_99];
       end
       function obj = ComparePreTResults(obj,results,parameter,names)
            XAxis =[];
            if parameter == 'LR'
                XAxis = obj.LearningRates;
            elseif parameter =='MC'
                XAxis = obj.Momentums;
            elseif parameter =='EP'
                XAxis = obj.Num_Epochs;
            end

            figure(1);
            t1 = tiledlayout(3,3,'TileSpacing','tight','Padding','tight');
            for r = 1:size(results,1)
                perfTrain =[];
                perfVal   =[];
                perfTest  =[];
    
                for i = results(r,:)
                    perfTrain(end+1) = obj.PreTrainNetworks(i).ResultTable.Performance(1);
                    perfVal(end+1)   = obj.PreTrainNetworks(i).ResultTable.Performance(2);
                    perfTest(end+1)  = obj.PreTrainNetworks(i).ResultTable.Performance(3);
                    
                end
                nexttile;
                plot(XAxis,perfTrain*100,'r-o', ...
                    XAxis,perfVal*100,'g-+', ...
                    XAxis,perfTest*100,'b-*');
                xlabel(parameter);
                ylabel('Performance');
                title(sprintf('%s',names(r)));
                legend({'Train','Validation','Test'},'Location','best');
                hold on;
            end
            title(t1,sprintf('Performance at different %s ',parameter));
            saveas(gca,sprintf('PreTrainResults/Diff-%s-Performance.jpg',parameter));
            close all;

            figure(2);
            t2 = tiledlayout(3,3,'TileSpacing','tight','Padding','tight');
            for r = 1:size(results,1)
                accTrain  =[];
                accVal    =[];
                accTest   =[];
                for i = results(r,:)
                    accTrain(end+1)  = obj.PreTrainNetworks(i).ResultTable.Accuracy(1);
                    accVal(end+1)    = obj.PreTrainNetworks(i).ResultTable.Accuracy(2);
                    accTest(end+1)   = obj.PreTrainNetworks(i).ResultTable.Accuracy(3);
                end
                nexttile;
                plot(XAxis,accTrain*100,'r-o', ...
                    XAxis,accVal*100,'g-+', ...
                    XAxis,accTest*100,'b-*');
                xlabel(parameter);
                ylabel('Accuracy');
                title(sprintf('%s',names(r)));
                legend({'Train','Validation','Test'},'Location','best');
                hold on;
            end
            title(t2,sprintf('Accuracy at different %s ',parameter));
            saveas(gca,sprintf('PreTrainResults/Diff-%s-Accuracy.jpg',parameter));
            close all;
       end
       function obj = ShowPreTResults(obj)
            obj = ComparePreTResults(obj,obj.PreTDiffParam.LR,'LR',obj.PreTParamNames.LR(:,1));
            obj = ComparePreTResults(obj,obj.PreTDiffParam.MC,'MC',obj.PreTParamNames.MC(:,1));
            obj = ComparePreTResults(obj,obj.PreTDiffParam.EP,'EP',obj.PreTParamNames.EP(:,1));
       end

       function obj = Setup(obj)
           netindex = 0;
           obj.SameParam = struct('FCN1',[],'FCN2',[], ...
                'HU50',[],'HU100',[],'HU200',[], ...
                'DR1',[],'DR2',[],'DR3',[]);
           obj.DiffParam = struct('FCN',[],'HU',[],'DR',[]);
           obj.ParamNames = struct('FCN',[],'HU',[],'DR',[]);
            obj.ParamNames.FCN = ["HU50-DR1";"HU50-DR2";"HU50-DR3";"HU100-DR1";"HU100-DR2";"HU100-DR3";"HU200-DR1";"HU200-DR2";"HU200-DR3"];
            obj.ParamNames.HU = ["FCN1-DR1";"FCN1-DR2";"FCN1-DR3";"FCN2-DR1";"FCN2-DR2";"FCN2-DR3"];
            obj.ParamNames.DR =  ["FCN1-HU50";"FCN1-HU100";"FCN1-HU200";"FCN2-HU50";"FCN2-HU100";"FCN2-HU200"];
           for transferFcnIdx = 1:obj.Functions.size(1)
                for numHiddenLayer = obj.Num_HiddenLayerUnits
                    for divideRatio = 1:length(obj.Divide_Ratios)
                        netindex = netindex+1;
                        if obj.logLevel >=1
                            fprintf('\nSetup Train Network:========================> \n');
                        end
                        if obj.logLevel >= 2
                            fprintf('HyperParameters:\n\tLayer [1] TransferFcn=[%s]\n\tLayer [2] TransferFcn=[%s]\n\tCost Function [%s]\n\t[%d] Hidden Units\n\tDivide Into [TrainSet=%d, ValidateSet=%d, TestSet=%d]\n' ...
                                ,obj.Functions(transferFcnIdx,1),obj.Functions(transferFcnIdx,2),obj.Functions(transferFcnIdx,3), ...
                                numHiddenLayer,obj.Divide_Ratios(divideRatio,1)*100,obj.Divide_Ratios(divideRatio,2)*100,obj.Divide_Ratios(divideRatio,3)*100);
                        end
                        net = SingleNetwork(obj.NetworkType,numHiddenLayer,obj.Functions(transferFcnIdx,1),obj.Functions(transferFcnIdx,2),obj.Functions(transferFcnIdx,3),obj.Divide_Ratios(divideRatio,1),obj.Divide_Ratios(divideRatio,2),obj.Divide_Ratios(divideRatio,3), ...
                            'trainscg',500,0.8,0.01,500);
                        desiredFolder = './Results/';
                        path = strcat(desiredFolder,net.network.name);
                        if ~exist(path, 'dir')
                            mkdir(path);
                        end
                        net.Path = path;
                        net.ID   = netindex;
                        net.Tag  = 'Train';
                        obj.Networks(netindex) = net;
                        
                        if transferFcnIdx == 1
                            obj.SameParam.FCN1(end+1) = netindex;
                        elseif transferFcnIdx == 2
                            obj.SameParam.FCN2(end+1) = netindex;
                        end

                        if numHiddenLayer == 50
                            obj.SameParam.HU50(end+1) = netindex;
                        elseif numHiddenLayer == 100
                            obj.SameParam.HU100(end+1) = netindex;
                        elseif numHiddenLayer == 200
                            obj.SameParam.HU200(end+1) = netindex;
                        end

                        if divideRatio == 1
                            obj.SameParam.DR1(end+1) = netindex;
                        elseif divideRatio == 2
                            obj.SameParam.DR2(end+1) = netindex;
                        elseif divideRatio == 3
                            obj.SameParam.DR3(end+1) = netindex;
                        end

                        if obj.logLevel >=1
                            fprintf('[%d]\tName: %s\n',netindex,obj.Networks(end).network.name);
                        end
                        
                    end
                end
           end
           obj = obj.GroupParam();
       end
       function obj = TrainAllWith(obj,Param, value)
            networkInd = [];
            for i = 1:length(obj.Networks)
                if Param == 'FCN'
                    if value == 1 & obj.Networks(i).network.transferFcn{2}=='logsig'
                        networkInd(end+1) = i;
                    elseif value ==2 & obj.Networks(i).network.transferFcn{2}=='softmax'
                        networkInd(end+1) = i;
                    end
                elseif Param == 'HU'
                    if value == size(obj.Networks(i).network.layers{1})
                        networkInd(end+1) = i;
                    end
                elseif Param == 'DR'
                    if value ==  obj.Networks(i).network.divideParam.trainRatio
                        networkInd(end+1) = i;
                    end
                end
            end
            for index = 1:length(networkInd)
               if obj.logLevel >=1
                fprintf('[%d/%d] ',index,length(networkInd));
               end
               obj = obj.Train(networkInd(index));
           end
           obj.TotalTrainTime = sum([obj.Networks(netwokInd).TotalTrainTime]);
           if obj.logLevel >=1
                fprintf('All Train Finished! Time: %.3f\n',obj.TotalTrainTime);
           end
       end
       function obj = TrainAll(obj)
           for index = 1:length(obj.Networks)
               if obj.logLevel >=1
                fprintf('[%d/%d] ',index,length(obj.Networks));
               end
               obj = obj.Train(index);
           end
           obj.TotalTrainTime = sum([obj.Networks.TotalTrainTime]);
           if obj.logLevel >=1
                fprintf('All Train Finished! Time: %.3f\n',obj.TotalTrainTime);
           end
       end
       function obj = Train(obj,index)
           if obj.logLevel >=1
                fprintf('Network Info: %s',obj.Networks(index).network.userdata.note); 
           end
           obj.Networks(index) = obj.Networks(index).Train(obj.Inputs,obj.Outputs,obj.TrainTimes);
       end
       function obj = GroupParam(obj)
            % FCN
            HU50_DR1 = [];
            HU50_DR2 = [];
            HU50_DR3 = [];
            HU100_DR1 = [];
            HU100_DR2 = [];
            HU100_DR3 = [];
            HU200_DR1 = [];
            HU200_DR2 = [];
            HU200_DR3 = [];

            % HU
            FCN1_DR1 =[];
            FCN1_DR2 =[];
            FCN1_DR3 =[];
            FCN2_DR1 =[];
            FCN2_DR2 =[];
            FCN2_DR3 =[];

            % DR
            FCN1_HU50 =[];
            FCN1_HU100 =[];
            FCN1_HU200 =[];
            FCN2_HU50 =[];
            FCN2_HU100 =[];
            FCN2_HU200 =[];
            
            for index = 1:length(obj.Networks)
                if any(obj.SameParam.FCN1(:) == index)
                    if any(obj.SameParam.HU50(:) == index)
                        FCN1_HU50(end+1) = index;
                    elseif any(obj.SameParam.HU100(:) == index)
                        FCN1_HU100(end+1) = index;
                    elseif any(obj.SameParam.HU200(:) == index)
                        FCN1_HU200(end+1) = index;
                    end
                    if any(obj.SameParam.DR1(:) == index)
                        FCN1_DR1(end+1) = index;
                    elseif any(obj.SameParam.DR2(:) == index)
                        FCN1_DR2(end+1) = index;
                    elseif any(obj.SameParam.DR3(:) == index)
                        FCN1_DR3(end+1) = index;
                    end
                elseif any(obj.SameParam.FCN2(:) == index)
                    if any(obj.SameParam.HU50(:) == index)
                        FCN2_HU50(end+1) = index;
                    elseif any(obj.SameParam.HU100(:) == index)
                        FCN2_HU100(end+1) = index;
                    elseif any(obj.SameParam.HU200(:) == index)
                        FCN2_HU200(end+1) = index;
                    end
                    if any(obj.SameParam.DR1(:) == index)
                        FCN2_DR1(end+1) = index;
                    elseif any(obj.SameParam.DR2(:) == index)
                        FCN2_DR2(end+1) = index;
                    elseif any(obj.SameParam.DR3(:) == index)
                        FCN2_DR3(end+1) = index;
                    end
                end
                if any(obj.SameParam.HU50(:) == index)
                    if any(obj.SameParam.DR1(:) == index)
                        HU50_DR1(end+1) = index;
                    elseif any(obj.SameParam.DR2(:) == index)
                        HU50_DR2(end+1) = index;
                    elseif any(obj.SameParam.DR3(:) == index)
                        HU50_DR3(end+1) = index;
                    end
                elseif any(obj.SameParam.HU100(:) == index)
                    if any(obj.SameParam.DR1(:) == index)
                        HU100_DR1(end+1) = index;
                    elseif any(obj.SameParam.DR2(:) == index)
                        HU100_DR2(end+1) = index;
                    elseif any(obj.SameParam.DR3(:) == index)
                        HU100_DR3(end+1) = index;
                    end
                elseif any(obj.SameParam.HU200(:) == index)
                    if any(obj.SameParam.DR1(:) == index)
                        HU200_DR1(end+1) = index;
                    elseif any(obj.SameParam.DR2(:) == index)
                        HU200_DR2(end+1) = index;
                    elseif any(obj.SameParam.DR3(:) == index)
                        HU200_DR3(end+1) = index;
                    end
                end

            end
            obj.DiffParam.FCN = [HU50_DR1;HU50_DR2;HU50_DR3;HU100_DR1;HU100_DR2;HU100_DR3;HU200_DR1 ;HU200_DR2; HU200_DR3];
            obj.DiffParam.HU = [FCN1_DR1;FCN1_DR2;FCN1_DR3;FCN2_DR1;FCN2_DR2;FCN2_DR3];
            obj.DiffParam.DR = [FCN1_HU50;FCN1_HU100;FCN1_HU200;FCN2_HU50;FCN2_HU100;FCN2_HU200];
            

       end
       function obj = CompareResult(obj,results,parameter,names)
            XAxis =[];
            if parameter == 'FCN'
                XAxis = obj.LearningRates;
            elseif parameter =='HU'
                XAxis = obj.Momentums;
            elseif parameter =='DR'
                XAxis = obj.Num_Epochs;
            end

            figure(1);
            t1 = tiledlayout(3,3,'TileSpacing','tight','Padding','tight');
            for r = 1:size(results,1)
                perfTrain =[];
                perfVal   =[];
                perfTest  =[];
    
                for i = results(r,:)
                    perfTrain(end+1) = obj.Networks(i).ResultTable.Performance(1);
                    perfVal(end+1)   = obj.Networks(i).ResultTable.Performance(2);
                    perfTest(end+1)  = obj.Networks(i).ResultTable.Performance(3);
                    
                end
                nexttile;
                plot(XAxis,perfTrain*100,'r-o', ...
                    XAxis,perfVal*100,'g-+', ...
                    XAxis,perfTest*100,'b-*');
                xlabel(parameter);
                ylabel('Performance');
                title(sprintf('%s',names(r)));
                legend({'Train','Validation','Test'},'Location','best');
                hold on;
            end
            title(t1,sprintf('Performance at different %s ',parameter));
            saveas(gca,sprintf('Results/Diff-%s-Performance.jpg',parameter));
            close all;

            figure(2);
            t2 = tiledlayout(3,3,'TileSpacing','tight','Padding','tight');
            for r = 1:size(results,1)
                accTrain  =[];
                accVal    =[];
                accTest   =[];
                for i = results(r,:)
                    accTrain(end+1)  = obj.Networks(i).ResultTable.Accuracy(1);
                    accVal(end+1)    = obj.Networks(i).ResultTable.Accuracy(2);
                    accTest(end+1)   = obj.Networks(i).ResultTable.Accuracy(3);
                end
                nexttile;
                plot(XAxis,accTrain*100,'r-o', ...
                    XAxis,accVal*100,'g-+', ...
                    XAxis,accTest*100,'b-*');
                xlabel(parameter);
                ylabel('Accuracy');
                title(sprintf('%s',names(r)));
                legend({'Train','Validation','Test'},'Location','best');
                hold on;
            end
            title(t2,sprintf('Accuracy at different %s ',parameter));
            saveas(gca,sprintf('Results/Diff-%s-Accuracy.jpg',parameter));
            close all;
       end
       function obj = ShowResults(obj)
            obj = CompareResult(obj,obj.DiffParam.FCN,'FCN',obj.ParamNames.FCN(:,1));
            obj = CompareResult(obj,obj.DiffParam.HU,'HU',obj.ParamNames.HU(:,1));
            obj = CompareResult(obj,obj.DiffParam.DR,'DR',obj.ParamNames.DR(:,1));
       end
   end
end