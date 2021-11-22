function r = CustomFWNN(inputs, outputs)
    if(nargin == 2)
        net = feedforwardnet(10);
        net = train(net,inputs,outputs);
        view(net);
        y = net(inputs);
        r = perform(net,y,outputs);
    end
end