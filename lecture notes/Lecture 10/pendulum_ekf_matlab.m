% Dynamics
function xdot = pendulum_dynamics(x, u)
    gravity = 9.81;
    length = 1.0;
    mass = 1.0;

    xdot = [x(2);
           -(gravity/length)*sin(x(1)) + u/(mass*length*length)];
end

function xn = pendulum_step(x, u)
    dt = 0.1;
    xm = x + 0.5*dt*pendulum_dynamics(x,u);
    xn = x + dt*pendulum_dynamics(xm,u);
end

% Dynamics Jacobians
function Ac = pendulum_deriv(x, u)
    gravity = 9.81;
    length = 1.0;
    mass = 1.0;

    Ac = [0                          1.0;
         -(gravity/length)*cos(x(1))  0];
end

function A = pendulum_step_deriv(x, u)
    dt = 0.1;
    xm = x + 0.5*dt*pendulum_dynamics(x,u);

    A = eye(2) + dt*pendulum_deriv(xm,u)*(eye(2) + 0.5*dt*pendulum_deriv(x,u));
end

% Measurements: we can see the tip of the pendulum
function y = tip(x)
    length = 1.0;
    y = [length*sin(pi-x(1)); length*cos(pi-x(1))];
end

function C = tip_deriv(x)
    length = 1.0;
    C = [-length*cos(pi-x(1)) 0;
         length*sin(pi-x(1))  0];
end

% Noise
V = 0.0001*eye(2); %process noise
W = 0.1*eye(2); %measurement noise

% Simulation
Tfinal = 10;
Nt = 101;
xtraj = zeros(2, Nt);
xtraj(:,1) = [2.5; 0];

ytraj = zeros(2, Nt);
ytraj(:,1) = tip(xtraj(:,1));

for k = 1:(Nt-1)
    xtraj(:,k+1) = pendulum_step(xtraj(:,k), 0) + sqrt(V)*randn(2,1);
    ytraj(:,k+1) = tip(xtraj(:,k+1)) + sqrt(W)*randn(2,1);
end

% Let's look at the state trajectory
times = linspace(0,Tfinal,Nt);
figure()
plot(times, xtraj(1,:), color='b')
hold on
plot(times, xtraj(2,:), color='r')

% Let's look at the measurements
figure()
plot(times, ytraj(1,:))
hold on
plot(times, ytraj(2,:))

% Filter Initialization
xfilt = zeros(2, Nt);
xfilt(:,1) = xtraj(:,1) + sqrt(W)*randn(2,1);

P = zeros(2,2,Nt);
P(:,:,1) = W;

for k = 1:(Nt-1)
    %Prediction
    xpred = pendulum_step(xfilt(:,k), 0);
    A = pendulum_step_deriv(xfilt(:,k), 0);
    Ppred = A*P(:,:,k)*A' + V;

    %Innovation
    z = ytraj(:,k+1) - tip(xpred);
    C = tip_deriv(xpred);
    S = C*Ppred*C' + W;

    %Kalman Gain
    L = Ppred*C'/S;

    %Update
    xfilt(:,k+1) = xpred + L*z;
    
    P(:,:,k+1) = (eye(2) - L*C)*Ppred*(eye(2) - L*C)' + L*V*L';
end

% Plot the estimate vs. the ground truth
figure()
plot(times, xtraj(1,:), color='b')
hold on
plot(times, xfilt(1,:), color='r')

figure()
plot(times, xtraj(2,:), color='b')
hold on
plot(times, xfilt(2,:), color='r')
