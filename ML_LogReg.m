%% Machine Learning Course (STANFORD) - Logistic Regression
%% Joan Cardona, 29/09/2015

%% In this exercise we will predict whether a student will be admitted
%% or not to a university given the scores of two exams.

%% The training data consists of m = 100 training examples of students
%% who have or have not been admitted to the university and their scores.
clear ; close all; clc

data = load('ex2data1.txt');

X = data(:, [1, 2]); % Exams 1 & 2 Scores
y = data(:, 3); % 1 - Admitted, 0- Not Admitted

%% Let's plot the data:
% Find Indices of Positive and Negative Examples
pos = find(y == 1); 
neg = find(y == 0);
% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, ...
'MarkerSize', 7);
hold on
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')
% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

%% Let's define the hypothesis: 
[m, n] = size(X); % m = 100 training examples and n = 2 features

% Let us add a full column of ones to X for convenience:
X = [ones(m, 1) X];

theta = zeros(n + 1, 1); % Initial theta, of size 3 x 1 (n + 1 features)

J = (- 1/m) .* (sum(y .* log(1./(1 + exp( - X * theta))) + (1 - y) .* log(1 - (1./(1 + exp( - X * theta)))))); 

aux_a = (1./(1 + exp(-(X * theta))) - y); % this is h - y

grad = (1/m) .* [sum(aux_a .* X(:,1)); sum(aux_a .* X(:,2)); sum(aux_a .* X(:,3));];

% J is the cost function we want to minimize
% grad is a length 3 vector of the values we have to substract from theta
% every time we upgrade the Gradient Descent algorithm (SIMULTANEOUSLY!!!)

% I'm going to save both the cost and the gradient parameters in a 
% vector that we will use later to minimize it tweaking theta:

%%%%%%%%costAndGrad = [J grad];

% In this case, we will not be using Gradient Descent, but we will 
% use the so-called fminunc function. Code copies from ex2.m:

% ----

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta, feeding it the cost
% and the gradient...
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), theta, options);

plotDecisionBoundary(theta, X, y);
% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;
% ----

% Couple of examples

student_case = [1 45 85];
prob = (1/(1 + exp(-((student_case * theta)))));
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n\n'], prob);
     
student_case_2 = [1 100 85];
prob_2 = (1/(1 + exp(-((student_case_2 * theta)))));
fprintf(['For a student with scores 100 and 85, we predict an admission ' ...
         'probability of %f\n\n'], prob_2);
 
%% END of script
 
