function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sum = 0;
for i = 1:m,
	ht =  sigmoid(X(i,:)*theta);
	sum = sum + y(i) * log(ht) + (1-y(i)) * log(1 - ht);
end;
sumt = 0;

for j = 2:n,
	sumt = sumt + theta(j)^2;
end;

J = -1 * sum/m + lambda/(2*m) * sumt;

grad = ((sigmoid(X*theta)-y)'*X)'/m;

for j=2:n,
	grad(j) = grad(j) + lambda/m * theta(j)
end;

% =============================================================

end
