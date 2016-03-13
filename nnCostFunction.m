function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
size_Theta1 = size(Theta1);

size_Theta2 = size(Theta2);

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


size_nn = size(nn_params);

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
ones = ones(size(X,1),1);
size_ones = size(ones);
size_X = size(X);
X_bias = [ones,X];
a_2 = X_bias * Theta1';
g_a_2 = 1 ./ (1+exp(-a_2));
g_a_2_bias = [ones,g_a_2];
size_g_a_2_bias = size(g_a_2_bias);

a_3 = g_a_2_bias * Theta2';
g_a_3 = 1 ./ (1+exp(-a_3));
[x ix] = max(g_a_3,[],2);
h = g_a_3;
h_output = ix;
size_h_output = size(h_output);

sum_vector = zeros(m,1);

for i=1:m

%cost calculation start here
part_1 = zeros(num_labels,1);
part_2 = zeros(num_labels,1);
part_3 = zeros(num_labels,1);
digit = y(i);
y_vector = zeros(num_labels,1);
y_vector(digit) = 1;
h_row = h(i,:);
h_row_vector = h_row';

ones_vector = ones(num_labels,1);
one_minus_h_row_vector = ones_vector - h_row_vector;
one_y_vector = ones_vector - y_vector;
part_1 = y_vector .* log(h_row_vector);
part_2 = one_y_vector .* log(one_minus_h_row_vector);
part_3 = part_1 + part_2;
sum_of_all = sum(part_3);
sum_vector(i) = sum_of_all;
%cost calculation ends here


end

sumOfSumVector = sum(sum_vector);
cost = (-1/m) * sumOfSumVector;
J_unregularized = cost;


regularization_cost_matrix_Theta1 = (Theta1(:,2:size_Theta1(2)));
Theta_1_unbiased_vector = regularization_cost_matrix_Theta1(:) .* regularization_cost_matrix_Theta1(:);
sum_theta1 = sum(Theta_1_unbiased_vector);
regularization_cost_matrix_Theta2 = (Theta2(:,2:size_Theta2(2)));
Theta_2_unbiased_vector = regularization_cost_matrix_Theta2(:) .* regularization_cost_matrix_Theta2(:);
sum_theta2 = sum(Theta_2_unbiased_vector);
sum_regularized = (lambda)*(sum_theta1 + sum_theta2)/(2*m);
J = J_unregularized + sum_regularized;



for t=1:m

a_1 = X(t,:);
b = a_1';
a_1 = [1;b];
z_2 = a_1' * Theta1';
a_2 = 1 ./ (1+ (exp(-z_2)));
a_2 = [1;a_2'];

z_3 = a_2' * Theta2';
a_3 = 1 ./ (1+ (exp(-z_3)));
digit = y(t);
y_vector = zeros(num_labels,1);
y_vector(digit) = 1;
delta_3 = a_3 - y_vector';
z_der_2 = a_2 .* (1 - a_2);
delta_2 = (Theta2' * delta_3') .* (z_der_2);
Theta2_grad = Theta2_grad + (delta_3' * a_2');
delta_2 = delta_2(2:end);

Theta1_grad = Theta1_grad + (delta_2 * a_1');


end
Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;








% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
