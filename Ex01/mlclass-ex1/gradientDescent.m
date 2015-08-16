function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    temp1=0;
    temp2=0;
    %compute the h just like costFunction
    for i = 1:m
        temp1 = temp1 + (theta' * X(i,:)' - y(i));
        temp2 = temp2 + (theta' * X(i,:)' - y(i)) * X(i,2);
    end
    
    %compute the theta using gradientDescent
    temptheta1 = theta(1) - (alpha/m)*temp1;
    temptheta2 = theta(2) - (alpha/m)*temp2;

    %have to simultaneously
    theta(1)=temptheta1;
    theta(2)=temptheta2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
end
