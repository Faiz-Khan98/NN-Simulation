clear
clc

load cancer_dataset.mat
inputs = cancerInputs ;
targets = cancerTargets ;

nodes = [2 8 32];
epochs = [4 8 16];
num_classifiers = [1 3 9 15];
iter = 30;
optimisers = ["trainscg", "trainlm", "trainrp"];

std_train = zeros(1, length(num_classifiers));
std_test = zeros(1, length(num_classifiers));
mean_train = zeros(1, length(num_classifiers));
mean_test = zeros(1, length(num_classifiers));

for i = 1: length(num_classifiers)
    test_error = zeros(1, iter );
    train_error = zeros(1, iter );
    
    net = patternnet(8,"trainscg");
    net.trainParam.epochs = 8;
    
    for l=1: iter
        nets = {};
        net.divideParam.trainRatio = 50/100;
        net.divideParam.valRatio = 0;
        net.divideParam.testRatio = 50/100;
        net.trainParam.showWindow = false ;
        for j = 1: num_classifiers(i)
            net = init(net);
            [net , ttrain ] = train(net,inputs,targets);
            nets{j} = net ;
        end
    
        tgt = targets(1 ,:);
        trTarg = [tgt(ttrain.trainInd);1- tgt(ttrain.trainInd)];
        tsTarg = [tgt(ttrain.testInd);1- tgt(ttrain.testInd)];
    
        out = zeros(1, length(tgt));
        for j = 1: num_classifiers(i)
            net = nets{j};
            outs = net(inputs);
            out = out + round(outs(1 ,:));
        end
        
        out = (sign(out/num_classifiers(i) - 0.5) /2) + 0.5;

        trOut = [out(ttrain.trainInd); 1- out(ttrain.trainInd)];
        tsOut = [out(ttrain.testInd); 1-out(ttrain.testInd)];
        [train_error(l),a,b,c] = confusion( trTarg , trOut );
        [test_error(l),a,b,c] = confusion( tsTarg , tsOut );
    end
    
    mean_train(i) = mean( train_error );
    std_train(i) = std( train_error );
    mean_test(i) = mean( test_error );
    std_test(i) = std( test_error );
end

figure (1)
hold on
plot(num_classifiers , mean_train ,'r-+','linewidth' ,2)
plot(num_classifiers , mean_test ,'b-+','linewidth' ,2)
plot(num_classifiers , std_train ,'m:+','linewidth' ,2)
plot(num_classifiers , std_test ,'c:+','linewidth' ,2)
legend('Training- Mean Error Rate ','Test- Mean Error Rate ','Training- Std','Test- Std')
xlabel(' Classifiers ')
ylabel('Error')

title ('8 Nodes - 8 Epochs ')
axis ([0 30 0 0.06])
axis = gca ;
axis . FontSize = 11;

% Changing nodes
mean_train = zeros(length(nodes), length(epochs));
std_train = zeros(length(nodes), length(epochs));
mean_test = zeros(length(nodes), length(epochs));
std_test = zeros(length(nodes), length(epochs));

for i = 1: length(nodes)
for j = 1: length(epochs)
train_error = zeros(1, iter);
test_error = zeros(1, iter);

net = patternnet(nodes(i),"trainscg");
net.trainParam.epochs =epochs(j);

for l=1: iter
nets = {};
net.trainParam.showWindow = false ;
net.divideParam.trainRatio = 50/100;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 50/100;

for k = 1:15
net = init(net);
[net ,a] = train(net ,inputs,targets);
nets{k} = net ;
end

[a, ttrain ] = train(net ,inputs,targets);

tgt = targets (1 ,:);
trTarg = [tgt(ttrain.trainInd);1- tgt(ttrain.trainInd)];
tsTarg = [tgt(ttrain.testInd);1- tgt(ttrain.testInd)];

out = zeros(1, length(tgt));

for k = 1:15
net = nets{k};
outs = net(inputs);
out = out + round(outs(1 ,:));
end

out = (sign( out /15 - 0.5) /2) + 0.5;

trOut = [out(ttrain.trainInd ); 1- out(ttrain.trainInd)];
tsOut = [out(ttrain.testInd ); 1-out(ttrain.testInd)];

[train_error(l),a,b,c] = confusion(trTarg , trOut);
[test_error(l),a,b,c] = confusion(tsTarg , tsOut);
end
mean_train(i,j) = mean( train_error );
std_train(i,j) = std( train_error );
mean_test(i,j) = mean( test_error );
std_test(i,j) = std( test_error );
end
end
xv = {'2', '8', '32'};
yv = {'16','8','4'};

figure (2)
axis = gca ;
axis.FontSize = 11;
set(findall(gcf ,'type','line'),'linewidth' ,2);

subplot (2 ,2 ,1)
h = heatmap(xv ,yv , flipud(mean_train),'Colormap',turbo);
h.XLabel = 'Nodes';
h.YLabel = 'Epochs';
h.Title = ' Training- Mean Error Rate ';

subplot (2 ,2 ,2)
h = heatmap (xv ,yv , flipud(mean_test),'Colormap',turbo);
h.XLabel = 'Nodes ';
h.YLabel = 'Epochs ';
h.Title = 'Test- Mean Error Rate ';

subplot (2 ,2 ,3)
h = heatmap (xv ,yv , flipud(std_train),'Colormap',turbo);
h.XLabel = 'Nodes ';
h.YLabel = 'Epochs ';
h.Title = 'Training- Std ';

subplot (2 ,2 ,4)
h = heatmap (xv ,yv , flipud(std_test),'Colormap',turbo);
h.XLabel = 'Nodes ';
h.YLabel = 'Epochs ';
h.Title = 'Test- Std ';

mean_train = zeros( length( optimisers ), length(nodes )* length(epochs));
std_train = zeros( length( optimisers ), length(nodes )* length(epochs));
mean_test = zeros( length( optimisers ), length( nodes )* length(epochs));
std_test = zeros( length( optimisers ), length( nodes )* length(epochs));

% Changing classifiers .
for s = 1: length( optimisers )
for i = 1: length( nodes )
for j = 1: length(epochs)
train_error = zeros(1, iter );
test_error = zeros(1, iter );


net = patternnet(nodes(i), optimisers(s));
net.trainParam.epochs =epochs(j);


for l=1: iter
nets = {};
net.divideParam.trainRatio = 0.5;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0.5;
net.trainParam.showWindow = false ;
for k = 1:15
net = init( net );
[net ,a] = train(net ,inputs,targets);
nets {k} = net ;
end

[~, ttrain ] = train(net ,inputs,targets);

tgt = targets(1 ,:);
trTarg = [tgt(ttrain.trainInd);1- tgt(ttrain.trainInd)];
tsTarg = [tgt(ttrain.testInd);1- tgt(ttrain.testInd)];

out = zeros(1, length(tgt));
for k = 1:15
net = nets{k};
outs = net(inputs);
out = out + round( outs(1 ,:));
end

out = sign( out /15 - 0.5) /2 + 0.5;

trOut = [out(ttrain.trainInd); 1- out(ttrain.trainInd)];
tsOut = [out(ttrain.testInd); 1-out(ttrain.testInd)];

[ test_error(l),a,b,c] = confusion( tsTarg , tsOut );
[ train_error(l),a,b,c] = confusion( trTarg , trOut );

end
mean_train(s, 3*i -3+ j) = mean( train_error );
std_train(s, 3*i -3+ j) = std( train_error );
mean_test(s, 3*i -3+j) = mean( test_error );
std_test(s, 3*i -3+j) = std( test_error );

end
end
end

figure (3)
for i = 1:9
subplot (3,3 ,i)
X = categorical({ 'TrainMean ','TestMean ','TrainStd ','TestStd '});
Y = [ mean_train(:,i),mean_test(:,i),std_train(:,i), std_test(:,i)]';
barh(X,Y)
%ylim ([0 , 0.09])
ll = string ( nodes( ceil (i /3) )) + ' Nodes '+ string(epochs(i- ceil (i /3) *3+3) ) + ' Epochs ';
title (ll)
end