function dxdt = BacksteppingClosedLoop(t,x)

% complete the following

% syms x1 x2 x3 z3 V phi u_feedback

% V = ;

% phi = ;

% may need to compute some derivatives here using diff command

% z3 = ;

% u_feedback = ;

% u = vpa(subs(u_feedback,[x1 x2 x3],[x(1) x(2) x(3)]));

% now close the control loop
% dxdt(1,:) = x(1)^2 - x(1)^3 + x(2);

% dxdt(2,:) = x(3);

% dxdt(3,:) = u;

%{
% my derivation below, but this is slow

syms x1 x2 x3 V2 V3 phi1 z2 phi2 z3 u_feedback;

x1dot = x1^2 - x1^3 + x2;
x2dot = x3;
x3dot = u_feedback;

phi1 = -x1^2 - x1;
phi1dot = diff(phi1) * x1dot;

z2 = x2 - phi1;
V2 = 1/2*x1^2 + 1/2*z2^2;

phi2 = -z2 - x1 + phi1dot;
phi2dot = diff(phi2, x1) * x1dot + diff(phi2, x2) * x2dot;

z3 = x3 - phi2;
V3 = V2 + 1/2*z3^2;

u_feedback = -z3 - z2 + phi2dot;

u = vpa(subs(u_feedback,[x1 x2 x3],[x(1) x(2) x(3)]));
%}

% then manually find-and-replace x1 with x(1) etc. symbols in text editor
u = - 3*x(1) - 2*x(2) - x(3) - x(3)*(2*x(1) + 2) - (2*x(1) + 1)*(- x(1)^3 + x(1)^2 + x(2)) ...
    - (- x(1)^3 + x(1)^2 + x(2))*(2*x(1) + 2*x(2) ...
    + (2*x(1) + 1)*(- 3*x(1)^2 + 2*x(1)) + 2*x(1)^2 - 2*x(1)^3 + 2) - 2*x(1)^2;

dxdt(1,:) = x(1)^2 - x(1)^3 + x(2);

dxdt(2,:) = x(3);

dxdt(3,:) = u;