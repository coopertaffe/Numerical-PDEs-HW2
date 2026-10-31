%% Hw 2 problem 1
% Numerical PDES
%% CODE FROM HW0
% Problem 4
clear;clc;close all;
% params
a = 0;
b = 2*pi;

% immersed boundary
a_theta = 0;
b_theta = 2*pi;

% delta function
x0=0;
y0=0;
alpha = 1.2;
delta = @(x,y) (1 / (2 * pi * alpha^2)) * exp(-((x - x0).^2 + (y - y0).^2) / (2 * alpha^2));

%vector of different step sizes or grid points
NV = 2.^(2:4); 
hV = zeros(size(NV)); 

Err_inf = zeros(size(NV));

for l=1:length(NV)
    N=NV(l);
    h = (b-a)/N;
    hV(l)=h;
    x = linspace(a,b,N+1);
    y = linspace(a,b,N+1);

    % immersed boundary
    N_ib = linspace(a_theta,b_theta,N);
    x_ib = pi + cos(N_ib);
    y_ib = pi + sin(N_ib);

%% Method of manufactured solutions
% u(x,y) = sinxcosy
u_exact = @(x,y) sin(2*x).*cos(2*y);
% f 
f = @(x,y) -8*sin(2*x).*cos(2*y);

% Number of interior grid points in each direction
n = (N-1)^2;

% matrix index to flat index (j slow, i fast)
idx = @(i,j) (j-1)*(N-1) + i;

% sparse matrix set up
row = [];
col = [];
val = [];

for i=1:N-1
    for j=1:N-1
        k=idx(i,j);

        %center
        row(end+1) = k;
        col(end+1) = k;
        val(end+1) = -4;

        % left neighbor
        if i > 1
            row(end+1) = k;
            col(end+1) = idx(i-1,j);
            val(end+1) = 1;
        end

        % right neighbor
        if i < N-1
            row(end+1) = k;
            col(end+1) = idx(i+1,j);
            val(end+1) = 1;
        end

        % below neighor
        if j > 1
            row(end+1) = k;
            col(end+1) = idx(i,j-1);
            val(end+1) = 1;
        end

        % above neighbor
        if j < N-1
            row(end+1) = k;
            col(end+1) = idx(i,j+1);
            val(end+1) = 1;
        end
    end
end

A = sparse(row,col,val,n,n);

%% setting up the RHS vec
RHS = zeros(n+N,1);

for j = 1:N-1
    for i = 1:N-1
        k = idx(i,j);

        xx = x(i+1);
        yy = y(j+1);

        RHS(k) = h^2*f(xx,yy);

        %% dirichlet boundary conditions
        %left u_i=a,j
        if i == 1
            BC = u_exact(a,yy);
            RHS(k) = RHS(k) - BC;
        end

        %right: u_i=b,j 
        if i == N-1
            BC = u_exact(b,yy);
            RHS(k) = RHS(k) - BC;
        end

        %above: u_i,j=b
        if j == N-1
            BC = u_exact(xx,b);
            RHS(k) = RHS(k) - BC;
        end

        %below: u_i,j=a
        if j == 1
            BC = u_exact(xx,a);
            RHS(k) = RHS(k) - BC;
        end

    end
end

%% Immersed boundary conditions
for i = 1:N
    k = n+i;
    RHS(k) = u_exact(x_ib(i),y_ib(i));
end

% Writing apply method for matrix J
apply_J = @(phi) J(phi,x,y,x_ib,y_ib,N,idx,delta);
%apply_M = @(phi) M(phi,n,A,apply_J);
apply_M = @(phi) [A*phi(1:n); apply_J(phi(1:n))];

%% GMRES
tol = .00001;
badtol = .1;
max_its = n;
U = gmres(apply_M,RHS,[],tol,max_its);

% Full grid, including boundary
U_grid = zeros(N+1, N+1);

% Put interior solution into the grid
for j = 1:N-1
    for i = 1:N-1
        k = idx(i,j);
        U_grid(i+1,j+1) = U(k);
    end
end

% Boundary conditions
% x = 0
for j = 0:N
% change this with changing u
    U_grid(1,j+1) = u_exact(a, y(j+1));
end

% x = 5
for j = 0:N
% change this with changing u
    U_grid(N+1,j+1) = u_exact(b, y(j+1));
end

% y = 0
for i = 0:N
% change this with changing u
    U_grid(i+1,1) = u_exact(x(i+1), a);
end

% y = 5
for i = 0:N
% change this with changing u
    U_grid(i+1,N+1) = u_exact(x(i+1), b);
end

[X,Y] = meshgrid(x,y);
U_exact = u_exact(X,Y);
Err_inf(l) = max(abs(U_grid' - U_exact), [], 'all');

end

%plotting
figure
loglog(NV, Err_inf, '-o', 'Color', [.267,.467,.667], 'LineWidth', 3, 'MarkerFaceColor', [.267,.467,.667],'MarkerSize',10)
hold on
shiftFactor = 0.5;   % how far below the curve to place the line (smaller = lower)

% --- reference line: slope 1, short segment near the middle of the data ---
p1 = 2;
idx1 = 1:4;                      % pick a short window of NVec to draw the line over
Nref1 = NV(idx1);
C1 = shiftFactor * Err_inf(idx1(1)) * Nref1(1)^p1;
ref1 = C1 ./ Nref1.^p1;
loglog(Nref1, ref1,'--' , 'Color',[.4,.8,.933], 'LineWidth', 3)


hold off
xlabel('Grid Size (N)')
ylabel('Relative Error')
legend('Err_{\infty}','Reference Slope: -2', 'Location', 'northeast')
grid on


% Writing apply method for matrix J
function sigma = J(phi,x,y,x_ib,y_ib,N,idx,delta)
    sigma = zeros(N,1);
    for k = 1:N
        for j = 1:N-1
            for i = 1:N-1
                dx = x_ib(k) - x(i);
                dy = y_ib(k) - y(j);
                sigma(k) = sigma(k) +  phi(idx(i,j))*delta(dx,dy);
            end
        end
    end
end
%combine A and J
% function y = M(phi,n,A,apply_J)
%     %split phi
%     phi1 = phi(1:n);

%     y = [A*phi1;apply_J(phi1)];
% end