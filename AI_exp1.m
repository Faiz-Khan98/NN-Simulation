clear
clc

load cancer_dataset.mat
inp = cancerInputs ;
targ = cancerTargets ;

nodes = [2 8 16 32];
trainFcn = 'trainscg';

epochs = [1 2 4 8 16 32 64];
iter = 30;

for node_num = 1: length(nodes)
    std_train = zeros(1,length(epochs));
    std_test = zeros(1,length(epochs));
    mean_train = zeros(1,length(epochs));
    mean_test = zeros(1,length(epochs));

    for epoch_num = 1: length(epochs)
        net.trainParam.showWindow = false ;
        net = patternnet(nodes(node_num), trainFcn);
        net.trainParam.epochs = epochs(epoch_num);

        net.divideParam.trainRatio = 50/100;
        net.divideParam.valRatio = 0;
        net.divideParam.testRatio = 50/100;

        for l = 1: iter
            net = init(net);
            [net,ttrain] = train(net,inp,targ);

            tgt = targ (1 ,:);
            testTarg = [tgt(ttrain.testInd); 1-tgt(ttrain.testInd)];
            trainTarg = [tgt(ttrain.trainInd); 1-tgt(ttrain.trainInd)];

            alloutput = net(inp);
            output = alloutput (1 ,:);
            trainOut = [output(ttrain.trainInd); 1-output(ttrain.trainInd)];
            testOut = [output(ttrain.testInd); 1-output(ttrain.testInd)];

            [er_train(l),a,b,c] = confusion(trainTarg,trainOut);
            [er_test(l),a,b,c] = confusion(testTarg,testOut);
            performance = perform(net,targ,alloutput);
        end
        
        mean_train(epoch_num) = mean(er_train);
        std_train(epoch_num) = std(er_train);
        mean_test(epoch_num) = mean(er_test);
        std_test(epoch_num) = std(er_test);
    end

figure(1)
subplot(2,2 ,node_num)
hold on

plot(epochs,mean_train,'r-+','linewidth',2)
plot(epochs,mean_test,'b-+','linewidth',2)
plot(epochs,std_train,'m:+','linewidth',2)
plot(epochs,std_test,'c:+','linewidth',2)
legend ('Training- Mean Error Rate ','Test- Mean Error Rate ','Training- Std','Test- Std')
xlabel ('Epochs')
ylabel ('Error')
t = 'Node = '+ string(nodes(node_num));
title (t)
ax = gca ;
ax.FontSize = 13;
axis([0 70 0 0.2])
end