load('Input.mat');
load('Output.mat');
network = CustomNetwork(Input,Output);
network = network.PreTrainAll(1);
network = network.ShowPreTResults();

network = network.TrainAll(1);
network = network.ShowAllResult();

% show the best result of Hiddent Unit is 200
network = network.GetBestResult('HU',200);