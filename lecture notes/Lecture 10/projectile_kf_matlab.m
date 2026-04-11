g = 9.8;
Ac = [0 1; 0 0];
cc = [0; -g];

%Create discrete-time dynamics model
dt = 0.1;
M = expm([Ac cc; 0 0 0]*dt);
A = M(1:2,1:2);
c = M(1:2,3);

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
    xtraj(:,k+1) = A*xtraj(:,k) + c + sqrt(V)*randn(2,1);
end

figure()
times = linspace(0,Tfinal,Nt);
plot(times, xtraj(1,:), color='b')
hold on
plot(times, xtraj(2,:), color='r')

%Generate noisy measurements
ytraj = zeros(Nt);
for k = 1:Nt
    ytraj(k) = xtraj(1,k) + sqrt(W)*randn();
end

figure()
plot(times, xtraj(1,:), color='b')
hold on
plot(times, ytraj, color='r')

%Define the measurement model
C = [1.0 0];

%Filter Initialization
xfilt = zeros(2,Nt);
xfilt(:,1) = xtraj(:,1) + randn(2,1);

P = zeros(2,2,Nt);
P(:,:,1) = 1.0*eye(2);

%Run filter on data
for k = 1:(Nt-1)
    %Prediction
    xpred = A*xfilt(:,k) + c;
    Ppred = A*P(:,:,k)*A' + V;

    %Innovation
    z = ytraj(k+1) - C*xpred;
    S = C*Ppred*C' + W;

    %Kalman gain
    L = Ppred*C'/S;

    %Update
    xfilt(:,k+1) = xpred + L*z;
    P(:,:,k+1) = (eye(2)-L*C)*Ppred*(eye(2)-L*C)' + L*W*L';
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