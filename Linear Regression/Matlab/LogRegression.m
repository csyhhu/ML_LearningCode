%Read in the data
x = load('ex2x.dat');
y = load('ex2y.dat');
% open a new figure window
figure 
plot(x, y, 'o');
ylabel('Height in meters')
xlabel('Age in years')
numSample = length(y); % store the number of training examples
x = [ones(numSample, 1), x]; % Add a column of ones to x

numFeature = 2; %There are only 2 features
theta = ones(numFeature,1); %initialize weight(theta) to all zeros

maxIter = 200; %iteration times
alpha = 0.07; %learing step

for i=1:maxIter
    output = x * theta;
    error = output - y;
    %temp = x' * error;
    theta = theta - alpha .* (1/numSample).* x' * error; %Pay attention here, we should use dot product otherwise it will get wrong because of MATLAB's grammer
end

%Draw the lines:
hold on % Plot new data without clearing old plot
plot(x(:,2), x*theta, '-') % remember that x is now a matrix with 2 columns
                           % and the second column contains the time info
legend('Training data', 'Linear regression')

%Plot the J of theta function
J_vals = zeros(100, 100);   % initialize Jvals to 100x100 matrix of 0's
theta0_vals = linspace(-3, 3, 100);
theta1_vals = linspace(-1, 1, 100);

for i = 1:length(theta0_vals)
	  for j = 1:length(theta1_vals)
          t = [theta0_vals(i); theta1_vals(j)];
          J_vals(i,j) = (0.5/numSample) .* (x * t - y)' * (x * t - y);
    end
end

% Plot the surface plot
% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1')
    