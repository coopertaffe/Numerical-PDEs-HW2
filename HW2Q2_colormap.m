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

N=32;
h = (b-a)/N;
x = linspace(a,b,N+1);
y = linspace(a,b,N+1);

% immersed boundary
NN = N;
N_ib = (0:NN-1)*(2*pi/NN);
%N_ib = linspace(a_theta,b_theta,NN);
x_ib = pi + cos(N_ib);
y_ib = pi + sin(N_ib);

% delta function
x0=0;
y0=0;
alpha = 1.2*h;
delta = @(x,y) (1 / (2 * pi * alpha)) * exp(-((x - x0).^2 + (y - y0).^2) / (2 * alpha));

%% Method of manufactured solutions
u_exact = @(x,y) cos(2*x).*sin(2*y);
% f 
f = @(x,y) -8*cos(2*x).*sin(2*y);

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
RHS = zeros(n+NN,1);

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
for i = 1:NN
    k = n+i;
    RHS(k) = u_exact(x_ib(i),y_ib(i));
end

% % Boundary contributions to interpolation
% for k = 1:NN
%     BC_interp = 0;
%     for j = 1:N+1
%         for i = 1:N+1
%             % Only include outer boundary points
%             if i == 1 || i == N+1 || j == 1 || j == N+1
%                 dx = x_ib(k) - x(i);
%                 dy = y_ib(k) - y(j);

%                 BC_interp = BC_interp + h^2*u_exact(x(i),y(j))*delta(dx,dy);
%             end
%         end
%     end
%     RHS(n+k) = RHS(n+k) - BC_interp;
% end


% Writing apply method for matrix J
apply_J = @(phi) J(phi,x,y,x_ib,y_ib,N,NN,idx,delta);
apply_S = @(Q) S(Q,x,y,x_ib,y_ib,N,NN,idx,delta);
apply_M = @(phi) [A*phi(1:n) - (h^2)*apply_S(phi(n+1:n+NN)); apply_J(phi(1:n))];

%% GMRES
tol = .1*(h^2);
max_its = n+NN;
U = gmres(apply_M,RHS,n,tol,1);


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

surf(X,Y,U_grid')

xlabel('x')
ylabel('y')
zlabel('u(x,y)')
shading interp
colorbar
colormap cool

U_exact = sin(X).*cos(Y);




% Writing apply method for matrix J
function sigma = J(phi,x,y,x_ib,y_ib,N,NN,idx,delta)
    sigma = zeros(NN,1);
    h = (x(2)-x(1));
    for k = 1:NN
        for j = 1:N-1
            for i = 1:N-1
                dx = x_ib(k)-x(i+1);
                dy = y_ib(k)-y(j+1);
                sigma(k) = sigma(k) +  (h^2)*phi(idx(i,j))*delta(dx,dy);
            end
        end
    end
end
% Now the spreading matrix
function sigma = S(Q,x,y,x_ib,y_ib,N,NN,idx,delta)
    sigma = zeros((N-1)^2,1);
    for k = 1:NN
        for j = 1:N-1
            for i = 1:N-1
                dx = x_ib(k)-x(i+1);
                dy = y_ib(k)-y(j+1);
                sigma(idx(i,j)) = sigma(idx(i,j)) +  Q(k)*delta(dx,dy);
            end
        end
    end
end
