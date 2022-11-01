clear	
		
C1 = mvnrnd([1 0] ,[1 0; 0 1] , 3300/ 2) ;	
C2 = mvnrnd([2 0] ,[4 0; 0 4] , 3300/ 2) ;	
		
inputs = [ C1 ; C2 ]';	
		
targets = [ones(size( C1 ,1) ,1) ,zeros(size( C1 ,1) ,1);zeros(size( C2 ,1) ,1) ,ones(size( C2 ,1),1)]';
trainFcn = 'trainscg';
figure(1)
plot( C1(: ,1) , C1(: ,2) ,'+')
hold on
plot( C2(: ,1) , C2(: ,2) ,'x')
axis equal
grid on

x1 =linspace(-4 , 6 , 300) ;
y1 =linspace(-4 , 4 , 300) ;


opt = zeros(length( x1 ),length( y1 ));
for iterate = 1:length( y1 )
    for j = 1:length( x1 )
        val1 = -1/2*log(det([1 0; 0 1]) ) - 1/2* transpose ([ y1( iterate ); x1( j) ] -[1;0]) * inv([1 0; 0 1]) *([ y1( iterate ); x1( j) ] -[1;0]) ;
        val2 = -1/2*log(det([4 0; 0 4]) ) - 1/2* transpose ([ y1( iterate ); x1( j) ] -[2;0]) * inv([4 0; 0 4]) *([ y1( iterate ); x1( j) ] -[2;0]) ;
        if val1 > val2
            opt( j, iterate ) = 1;
        end
    end
end

figure(1)
hold  on
contour( y1 , x1 , opt ,[0 ,1] ,'y','linewidth', 2)

y1 =linspace(-3 , 3 ,100) ;
x1 =linspace( -2.5 , 4 , 100) ;
z =zeros(length( x1 ),length( y1 ));

net = patternnet(8 ,"trainscg");
net.trainParam.epochs = 8;
nets = {};
net.divideParam.trainRatio = 300 / 3300 ;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 3000 / 3300 ;
net.trainParam.showWindow = false ;
for j = 1:8
net = init( net );
[ net , a] = train( net , inputs , targets );
nets{j} = net ;
end

for iterate = 1:length( y1 )
    for j = 1:length( x1 )
        outp = 0;
        for k = 1:5
            net = nets{ k};
            outps = net([ y1( iterate ); x1( j)]);
            outp = outp +round( outps(1 ,:) );
        end
        z( j, iterate ) =sign( outp /5 - 0.5) /2 + 0.5;
    end
end

figure(1)
hold on
contour( y1 , x1 , z ,[0 ,1] ,'c','linewidth', 1.5)
legend("C1","C2","BayesDecisionBoundary","NeuralNet")
axis([ -7 ,14 , -8 ,6])