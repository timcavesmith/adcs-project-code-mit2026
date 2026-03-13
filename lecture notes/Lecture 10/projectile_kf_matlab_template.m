g = 9.8;
Ac = [0 1; 0 0];
cc = [0; -g];

%Create discrete-time dynamics model
dt = 0.1;
A = %% Fill me in
c = %% Fill me in

%Noise covariances
V = diag([0.01; 0.01]);
W = 1.0;

%Initial conditions
x0 = randn(2,1);

%Simulate
Tfinal = 2.0;
Nt = 21;
xtraj = zeros(2,Nt);
xtraj(:,1) = x0;
for k = 1:(Nt-1)
    xtraj(:,k+1) = %% Fill me in
end

figure()
times = linspace(0,Tfinal,Nt);
plot(times, xtraj(1,:), color='b')
hold on
plot(times, xtraj(2,:), color='r')

%Generate noisy measurements
ytraj = zeros(Nt);
for k = 1:Nt
    ytraj(k) = %% Fill me in
end

figure()
plot(times, xtraj(1,:), color='b')
hold on
plot(times, ytraj, color='r')

%Define the measurement model
C = %% Fill me in

%Filter Initialization
xfilt = zeros(2,Nt);
xfilt(:,1) = xtraj(:,1) + randn(2,1);

P = zeros(2,2,Nt);
P(:,:,1) = 1.0*eye(2);

%Run filter on data
for k = 1:(Nt-1)
    %Prediction
    xpred = %% Fill me in
    Ppred = %% Fill me in

    %Innovation
    z = %% Fill me in
    S = %% Fill me in

    %Kalman gain
    L = %% Fill me in

    %Update
    xfilt(:,k+1) = %% Fill me in
    P(:,:,k+1) = %% Fill me in
end

%Plot the estimate vs. the ground truth
figure()
plot(times, xtraj(1,:), color='b')
hold on
plot(times, xfilt(1,:), color='r')
plot(times, ytraj, color='g')

figure()
plot(times, xtraj(2,:), color='b')
hold on
plot(times, xfilt(2,:), color='r')

%Plot covariance components
figure()
plot(times, squeeze(P(1,1,:)), color='b')
hold on
plot(times, squeeze(P(2,2,:)), color='r')