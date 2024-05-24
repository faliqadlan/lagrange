function [SS, xx] = DynamicEqSolver(Eq, q, Dq, ParamList, ParamVal, tspan, InitCnd)
% Author: Mansour Torabi
% Email: smtoraabi@ymail.com
%
% This function solves a system of dynamic equations.
%
% Inputs:
% Eq - The equations of motion
% q - The generalized coordinates
% Dq - The derivatives of the generalized coordinates
% ParamList - The list of parameters in the equations
% ParamVal - The values of the parameters
% tspan - The time span for which to solve the equations
% InitCnd - The initial conditions for the equations
%
% Outputs:
% SS - The state-space equations
% xx - The solution to the state-space equations

%% [1.1]: Convert Eq To State-Space Form:
% This section converts the given equations of motion into a state-space form.

N = length(Eq);

DDq = sym(zeros(1, N));
for ii = 1:N
    DDq(ii) = sym(['DD', char(q(ii))]);
end

% AA * X = BB;

AA = jacobian(Eq, DDq);
BB = -simplify(Eq - AA*DDq.');

%% [1.2]: Solve for DDq:
% This section solves for the second derivatives of the generalized coordinates.

DDQQ   = sym(zeros(N, 1));
DET_AA = det(AA);

for ii = 1:N   
    AAn       = AA;
    AAn(:,ii) = BB;
    DDQQ(ii)  = simplify(det(AAn) / DET_AA);
end

%% [1.3]: Form the State-Space Equations:
% This section forms the state-space equations.

SS = sym(zeros(N, 1));

for ii = 1:N
   SS (ii) = Dq(ii);
   SS (ii + N) = DDQQ(ii);   
end

%% [1.4]: Change Variables from q to x:
% This section changes the variables from the generalized coordinates and their derivatives to a new variable x.

Q = [q, Dq];
X = sym('x_',[1 2*N]);
SS = subs(SS, Q, X);

%% [2.1] Solving ODEs:
% This section solves the ordinary differential equations formed by the state-space equations.

syms t
% Preparation of SS Eq for ODE Solver: Creating Anonymous Fcn

SS_0 = subs(SS, ParamList,ParamVal);
SS_ode0 = matlabFunction(SS_0,'vars',{X, t});

SS_ode  = @(t, x)SS_ode0(x(1:2*N)',t);
[ts, xx] = ode45(SS_ode, tspan, InitCnd); 

end
