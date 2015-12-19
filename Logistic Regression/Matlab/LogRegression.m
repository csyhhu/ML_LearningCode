%Read in the data
x = load('ex4x.dat');
y = load('ex4y.dat');
numSample = length(y); % store the number of training examples
x = [ones(numSample, 1), x]; % Add a column of ones to x
% find returns the indices of the
% rows meeting the specified condition
pos = find(y == 1); neg = find(y == 0);
% Assume the features are in the 2nd and 3rd
% columns of x
figure;
plot(x(pos, 2), x(pos,3), '+'); hold on
plot(x(neg, 2), x(neg, 3), 'o')

%define sigmoid function
g = inline('1.0 ./ (1.0 + exp(-z))'); 
% Usage: To find the value of the sigmoid 
% evaluated at 2, call g(2)

% %Grad Method It is very strange that I can make it right though I check
% it lots of times. I will update it when I work it out
% maxIter = 2000;
% alpha = 0.07;
% numFeature = 2;
 theta = zeros(numFeature+1,1);
% for i=1:maxIter
%     output = g(x * theta);
%     error = y - output;
%     theta = theta + 1.0/numSample .* sum(x' * error);
% end

%Newton's Method
NewTownIter = 7;
J(i) = 0;
for i=1:NewTownIter
    z = x * theta;
    output = g(z);
    derivativeJ = 1/numSample .* x' * (output - y);
    Hessian = (1/numSample) .* x' * diag(output) * diag(1-output) * x;
    
    J(i) =(1/numSample)*sum(-y.*log(output) - (1-output).*log(1-output));
    theta = theta - Hessian\derivativeJ;
end

% Plot result
% Only need 2 points to define a line, so choose two endpoints
plot_x = [min(x(:,2))-2,  max(x(:,2))+2];
% Calculate the decision boundary line
plot_y = (-1./theta(3)).*(theta(2).*plot_x +theta(1));
plot(plot_x, plot_y)
legend('Admitted', 'Not admitted', 'Decision Boundary')
hold off

% Plot J
figure
plot(0:NewTownIter-1, J, 'o--', 'MarkerFaceColor', 'r', 'MarkerSize', 8)
xlabel('Iteration'); ylabel('J')