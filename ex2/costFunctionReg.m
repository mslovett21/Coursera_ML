function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



z      = X * theta;
h      = sigmoid(z);
y_tran = transpose(y);

cost         = -y_tran*log(h) - (1- y_tran)*log(1-h);
reg          = lambda/m;
theta_sq     = theta.^2;
reg_theta    = (reg/2)* theta_sq;
reg_theta(1) = 0;
sum_reg_theta= sum(reg_theta);
J            = cost/m + sum_reg_theta;

diff   = h - y;
X_tran = transpose(X);
grad = X_tran * diff;
grad = grad./m + reg*theta;  
grad(1) = grad(1) - reg*theta(1);

% =============================================================

end
