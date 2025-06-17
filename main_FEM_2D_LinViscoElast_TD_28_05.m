clc; clear all; close all;
set(0,'DefaultFigureWindowStyle','docked')
addpath(genpath('routines'))
addpath 'C:\Users\utente\Documents\MATLAB\Condensazione_Diego'%% MAT0 % per prova ma mi serve per simulazioni rapide... quindi riduco il taumax... 
addpath 'C:\Users\utente\Documents\MATLAB\Material_characterization\EUCLID'


% G     = [1.019E+3 5.290E+2]'*1e4;
% tauG  = [4.824E-1 3.915E+0]'; % Devono essere messi sempre in ordine crescente
% Ginf = 5E2*1e4; % assume the same Einf for each element of the bar.
% NMeG  = length(G);
% G0 = Ginf + sum(G);
% g = G/G0; % Abaqus form
% 
% % Volumetric
% K     = [2.366E+3 1.097E+3]'*1e4;
% tauK  = [4.570E-1 4.197E+0]'; 
% Kinf = 2E3*1e4;
% NMeK  = length(K);
% K0 = Kinf + sum(K);
% k = K/K0; % Abaqus form

%% MAT1
% G     = [1.019E+3 5.290E+2 2.011E+2]';
% tauG  = [4.824E-1 3.915E+0 3.021E+1]'; % Devono essere messi sempre in ordine crescente
% Ginf = 5E2; % assume the same Einf for each element of the bar.
% NMeG  = length(G);
% G0 = Ginf + sum(G);
% g = G/G0; % Abaqus form
% 
% % Volumetric
% K     = [2.366E+3 1.097E+3]';
% tauK  = [4.570E-1 4.197E+0]'; 
% Kinf = 2E3;
% NMeK  = length(K);
% K0 = Kinf + sum(K);
% k = K/K0; % Abaqus form

%% MAT2
% G     = [1.019E+3 5.290E+2]';
% tauG  = [4.824E-1 3.915E+0]'; % Devono essere messi sempre in ordine crescente
% Ginf = 5E2; % assume the same Einf for each element of the bar.
% NMeG  = length(G);
% G0 = Ginf + sum(G);
% g = G/G0; % Abaqus form
% 
% % Volumetric
% K     = [2.366E+3]';
% tauK  = [4.570E-1]'; 
% Kinf = 2E3;
% NMeK  = length(K);
% K0 = Kinf + sum(K);
% k = K/K0; % Abaqus form

%% MAT3
% G     = [7.790E+02 1.019E+3 5.290E+2 2.011E+2 9.600E+1]';
% tauG  = [7.280E-02 4.824E-1 3.915E+0 3.021E+1 6.294E+2]'; % Devono essere messi sempre in ordine crescente
% Ginf = 5E2; % assume the same Einf for each element of the bar.
% NMeG  = length(G);
% G0 = Ginf + sum(G);
% g = G/G0; % Abaqus form
% 
% % Volumetric
% K     = [2.242E+03 2.712E+03 2.366E+3 1.097E+3 4.601E+02]';
% tauK  = [7.693E-03 6.344E-02 4.570E-1 4.197E+0 3.512E+01]';
% Kinf = 2E3;
% NMeK  = length(K);
% K0 = Kinf + sum(K);
% k = K/K0; % Abaqus form
%% MAT3.5
G     = [1000 2000 1234]';
tauG  = [ 5.327  50.125 300.2268]'; % Devono essere messi sempre in ordine crescente
Ginf = 5E2; % assume the same Einf for each element of the bar.
NMeG  = length(G);
G0 = Ginf + sum(G);
g = G/G0; % Abaqus form

% Volumetric
K     = [ 5000 7000 5678 ]';
tauK  = [ 5.327  50.125 300.2268]';
Kinf = 2E3;
NMeK  = length(K);
K0 = Kinf + sum(K);
k = K/K0; % Abaqus form


%% MAT4
% G     = [7.790E+02 1.019E+3 5.290E+2 2.011E+2 9.600E+1]';
% tauG  = [7.280E-02 4.824E-1 3.915E+0 3.021E+1 6.294E+2]'; % Devono essere messi sempre in ordine crescente
% Ginf = 5E2; % assume the same Einf for each element of the bar.
% NMeG  = length(G);
% G0 = Ginf + sum(G);
% g = G/G0; % Abaqus form
% 
% % Volumetric
% K     = [2.712E+03 2.366E+3 1.097E+3 4.601E+02 1.277E-03]';
% tauK  = [6.344E-02 4.570E-1 4.197E+0 3.512E+01 6.016E+02]';
% Kinf = 2E3;
% NMeK  = length(K);
% K0 = Kinf + sum(K);
% k = K/K0; % Abaqus form

%% MAT5
% G     = [7.790E+02 1.019E+3 5.290E+2 2.011E+2]'*1e4;
% tauG  = [7.280E-02 4.824E-1 3.915E+0 3.021E+1]'; % Devono essere messi sempre in ordine crescente
% Ginf = 5E2*1e4; % assume the same Einf for each element of the bar.
% NMeG  = length(G);
% G0 = Ginf + sum(G);
% g = G/G0; % Abaqus form
% 
% % Volumetric
% K     = [2.712E+03 2.366E+3 1.097E+3 4.601E+02]'*1e4;
% tauK  = [6.344E-02 4.570E-1 4.197E+0 3.512E+01]';
% Kinf = 2E3*1e4;
% NMeK  = length(K);
% K0 = Kinf + sum(K);
% k = K/K0; % Abaqus form

%% SET TARGET MATERIAL 
target_mat.G = G;
target_mat.tauG = tauG;
target_mat.Ginf = Ginf;%

target_mat.K = K;
target_mat.tauK = tauK;
target_mat.Kinf = Kinf;


%% Projectors in 2D for plane strain problm
m = [1 1 1 0]';
PrD =  eye(4) - 1/3*(m*m');
PrV =1/3*(m*m');
Dmu = [2 0 0 0;
    0 2 0 0;
    0 0 2 0;
    0 0 0 1];

%% Geometry
Lx = 40;%  5; % Lenght in x direction
nx = 11; %40;  % Number of nodex in x, referred to only on one half of the structure since here symmetry is exploited.
Ly = 60; %6;  % Lenght in y direction
ny = 31; %30;  % Number of nodex in y
NN = nx*ny; % Ttotal number of nodes
q0=50;    % Carico distribuito
% nx = 4;
%         ny = 31;
%% Nodes and coordinates
coord = zeros(NN,2);
kk = 1;
for i = 1:ny
    y = Ly/(ny-1)*(i-1);
    for j = 1:nx
        nn = (i-1)*nx + j;
        %         x = Lx/(nx-1)*(j-1);
        x = (Lx/2)/(nx-1)*(j-1); % due to symmetry
        coord(kk,1:3) = [nn,x,y];
        kk = kk + 1;
    end
end

id=(1:1:size(coord,1))';
x=coord(:,2);
y=coord(:,3);
verticalGDL=zeros(NN,1);
horizontalGDL=zeros(NN,1);

horizontalGDL(find(coord(:,2)==0))=4*ones(size(find(coord(:,2)==0)));
horizontalGDL(find(coord(:,2)==Lx/2))=3*ones(size(find(coord(:,2)==Lx/2)));

verticalGDL(find(coord(:,3)==0))=2*ones(size(find(coord(:,3)==0)));
verticalGDL(find(coord(:,3)==Ly))=ones(size(find(coord(:,3)==Ly)));

coord_tab = table(id,x,y,verticalGDL,horizontalGDL);
% Elements connectivity
conneL = zeros((nx-1)*(ny-1),3); % lower triangle
conneU = zeros((nx-1)*(ny-1),3); % upper triangle
ie = 1;
for i = 1:ny-1
    for j = 1:nx-1
        if round(ie/2,0)==round(ie/2,1)
            % Elementi pari
            conneL(ie,1) = (i-1)*nx + j;
            conneL(ie,2) = (i-1)*nx + j+1;
            conneL(ie,3) = i*nx + j;
            conneU(ie,1) = (i-1)*nx + j + 1;
            conneU(ie,2) = i*nx + j + 1;
            conneU(ie,3) = i*nx + j;
            ie = ie+1;
        else
            % Elementi dispari
            conneL(ie,1) = (i-1)*nx + j;
            conneL(ie,2) = (i-1)*nx + j + 1;
            conneL(ie,3) = i*nx + j + 1;
            conneU(ie,1) = (i-1)*nx + j;
            conneU(ie,2) = i*nx + j + 1;
            conneU(ie,3) = i*nx + j;
            ie = ie+1;
        end

    end
end
conne = [conneL;conneU];

% Strip of Bottom elements (say at y = 0)
ie = 1;
i = 1;
for j = 1:nx-1
    if round(ie/2,0)==round(ie/2,1)
        conneBL(ie,1) = (i-1)*nx + j;
        conneBL(ie,2) = (i-1)*nx + j + 1;
        conneBL(ie,3) = i*nx + j;
        ie = ie+1;
    else
        conneBL(ie,1) = (i-1)*nx + j;
        conneBL(ie,2) = (i-1)*nx + j + 1;
        conneBL(ie,3) = i*nx + j + 1;

        ie = ie+1;
    end
end
conneB = [conneBL];

% Strip of Top elements (say at y = Ly)
ie = 1;
i = ny-1;
for j = 1:nx-1
    if round(ie/2,0)==round(ie/2,1)
        conneTU(ie,1) = (i-1)*nx + j+1;
        conneTU(ie,2) = i*nx + j + 1;
        conneTU(ie,3) = i*nx + j;
        ie = ie+1;
    else
        conneTU(ie,1) = (i-1)*nx + j;
        conneTU(ie,2) = i*nx + j + 1;
        conneTU(ie,3) = i*nx + j;
        ie = ie+1;
    end
end
conneT = conneTU;

% Strip of Left elements (say at x = 0)
ie = 1;
for i = 1:ny-1
    j = 1;
    conneLU(ie,1) = (i-1)*nx + j;
    conneLU(ie,2) = i*nx + j + 1;
    conneLU(ie,3) = i*nx + j;
    ie = ie+1;
end
conneL = conneLU;

% Strip of Right elements (say at x = Lx)
ie = 1;
for i = 1:ny-1
    j = nx-1;
    conneRL(ie,1) = (i-1)*nx + j+1;
    conneRL(ie,2) = i*nx + j + 1;
    conneRL(ie,3) = i*nx + j ;
    ie = ie+1;
end
conneR = conneRL;


% Check num of elements
if size(conne,1) ~= 2*(nx-1)*(ny-1)
    return
end

% Plot initial mesh
figure
trimesh(conne,coord(:,2),coord(:,3),zeros(NN,1),'LineStyle','-','edgecolor',[.5 .5 ,.5],'FaceAlpha',0);
view([0 0 1])
grid on
axis equal
hold off
% for i=1:size(coord,1)
%   text(coord(i,2),coord(i,3),num2str(i));
%   hold on
% end

mesh_data.nx = nx;
mesh_data.ny = ny;
mesh_data.coord = coord;
mesh_data.conne = conne;
mesh_data.NN = NN;
save ('mesh_data','mesh_data', '-v7.3');
%
%% TD simulation settings
[ntimestep,Tsim,uc_th,time,dt] = TD_sim_setting_v03_inversePB(target_mat);
uc_th=uc_th*0;
%% Iniatialization t = 0
nt = 1;
t = dt*(nt-1);
KK = zeros(2*NN);
F = zeros(2*NN,1);
F2=F;
for ie = 1:size(conne,1)
    vrtx = conne(ie,1:3); % vertexes array
    X = [coord(vrtx(1),2),coord(vrtx(2),2),coord(vrtx(3),2)]';
    Y = [coord(vrtx(1),3),coord(vrtx(2),3),coord(vrtx(3),3)]';
    N_xi  = [-1 1 0]';
    N_eta = [-1 0 1]';
    Jmat(1,1) = N_xi'*X; Jmat(1,2) = N_xi'*Y;
    Jmat(2,1) = N_eta'*X; Jmat(2,2) = N_eta'*Y;
    
    %             betDnm1 = zeros(3,1);
    %             betVnm1 = zeros(3,1);
    
    % One Gauss point over the element
    xi_gp =  1/3; % not used actually
    eta_gp = 1/3; % not used actually
    w_gp = 0.5;
    detJ = det(Jmat);
    Ae(ie) = w_gp*detJ;
    Jm1 = 1/detJ*[Jmat(2,2), -Jmat(1,2); -Jmat(2,1), Jmat(1,1)];
    N1_x = Jm1(1,:)*[N_xi(1) N_eta(1)]';
    N1_y = Jm1(2,:)*[N_xi(1) N_eta(1)]';
    N2_x = Jm1(1,:)*[N_xi(2) N_eta(2)]';
    N2_y = Jm1(2,:)*[N_xi(2) N_eta(2)]';
    N3_x = Jm1(1,:)*[N_xi(3) N_eta(3)]';
    N3_y = Jm1(2,:)*[N_xi(3) N_eta(3)]';
    
    B = [N1_x 0    N2_x 0    N3_x 0;
        0     N1_y 0    N2_y 0    N3_y;
        0     0    0    0    0    0;
        N1_y  N1_x N2_y N2_x N3_y N3_x];
    BD = PrD*B;
    b = m'*B;
    
    
    %         BV = PrV*B;
    
    % Element matrix (no summation since yhere is only 1 GP)
    %         ke = ( BD'*( Ginf + sum(G) - G'*(dt./(2*tauG + dt)) )*BD + ...
    %                b'* ( Kinf + sum(K) - K'*(dt./(2*tauK + dt)) )*b )*w_gp*detJ;
    
    
    keG = ( Ginf + sum(G) )*B'*Dmu*BD*w_gp*detJ;
    keK = ( Kinf + sum(K) )*(b'*b)*w_gp*detJ;
    ke = keG + keK;
    
    febetnm1 = zeros(6,1);
    
    
    % Assembly
    for i = 1:3
        I = conne(ie,i);
        for j = 1:3
            J = conne(ie,j);
            KK(2*(I-1)+1:2*(I-1)+2,2*(J-1)+1:2*(J-1)+2) = KK(2*(I-1)+1:2*(I-1)+2,2*(J-1)+1:2*(J-1)+2) +...
                ke(2*(i-1)+1:2*(i-1)+2,2*(j-1)+1:2*(j-1)+2);
        end
        F(2*(I-1)+1:2*(I-1)+2,1) = F(2*(I-1)+1:2*(I-1)+2,1) +...
            febetnm1(2*(i-1)+1:2*(i-1)+2,1);
        F2(2*(I-1)+1:2*(I-1)+2,1) = F2(2*(I-1)+1:2*(I-1)+2,1) +...
            febetnm1(2*(i-1)+1:2*(i-1)+2,1);
    end
end
KKold = KK;

% Boundary conditions: for the benchmark pb. (see Felippa §27.3.1. p. 27-4) and other simple cases

%% Bottom elements at y = 0;
kk = 1;
i = 1;
for j = 1:nx
    ndof(kk) = (i-1)*nx + j;
    kk = kk + 1;
end
dof_bottomX = 2*(ndof-1)+1;
dof_bottomY = 2*ndof;
dofB = [dof_bottomX,dof_bottomY];
% Clamped case
for k = 1:length(dof_bottomY)
    KK(dof_bottomY(k),:) = zeros(1,2*NN);
    KK(dof_bottomY(k),dof_bottomY(k)) = 1;
end
F(dof_bottomY) = 0;
for k = ceil(nx/2)
    KK(dof_bottomX(k),:) = zeros(1,2*NN);
    KK(dof_bottomX(k),dof_bottomX(k)) = 1;
end
F(dof_bottomX) = 0;


clear ndof

%% Lateral elements at x = 0 = symmetry axis!
kk = 1;
for i = 1:ny
    j = 1;
    ndof(kk) = (i-1)*nx + j;
    kk = kk + 1;
end
dofLx = 2*(ndof-1)+1;
% for k = 1:length(dofLx)
%     KK(dofLx(k),:) = zeros(1,2*NN);
%     KK(dofLx(k),dofLx(k)) = 1;
% end
% F(dofLx) = 0;

dof_leftX=dofLx(2:end);
dof_leftY=dofLx(2:end)+1;
dof_leftY(end)=[];
clear ndof

%% Lateral elements at x = Lx/2;

    kk = 1;
    for i = 1:ny
        j = nx;
        ndof(kk) = (i-1)*nx + j;
        kk = kk + 1;
    end
    dof_rightX = 2*ndof-1;
    dof_rightY = 2*ndof;
    % for k = 1:length(dofoffRx)
    %     K(dofoffRx(k),:) = zeros(1,2*NN);
    %     K(dofoffRx(k),dofoffRx(k)) = 1;
    % end
    clear ndof
%}

%% 247op elements at y = Ly;
% Unit displacement at beam top
% kk = 1;
% i = ny;
% for j = 1:nx
%     ndof(kk) = (i-1)*nx + j;
%     kk = kk+1;
% end
% %     dofTx = 2*(ndof-1)+1;
% dofTy = 2*ndof;
% % Unit displacement enforced
% for k = 1:length(dofTy)
%     KK(dofTy(k),:) = zeros(1,2*NN);
%     KK(dofTy(k),dofTy(k)) = 1;
% end
% F(dofTy) = uc_th(nt); %.1; % Unit displacement enforced
% clear ndof

% Distributed load on top elements at y = Ly;
%}
% Concentrated load at beam top

    kk = 1;
    i = ny;
    for j = 1:nx
        ndof(kk) = (i-1)*nx + j;
        kk = kk+1;
    end
    %     dofTx = 2*(ndof-1)+1;
   
    dof_topX = 2*ndof-1;
    dof_topY = 2*ndof;
    % F0 = 1000;
    % F(dofTy) = F0/length(dofTy);
    clear ndof
    % Distributed load on top elements at y = Ly;
%}

    for ie = 1:size(conneT,1)
        vrtx = conneT(ie,1:3);
        X = [coord(vrtx(1),2),coord(vrtx(2),2),coord(vrtx(3),2)]';
        Y = [coord(vrtx(1),3),coord(vrtx(2),3),coord(vrtx(3),3)]';
        %     jL1 = norm([X(2) Y(2)] - [X(1) Y(1)]);
        jL2 = norm([X(3) Y(3)] - [X(2) Y(2)]);
        %     jL3 = norm([X(3) Y(3)] - [X(1) Y(1)]);

        % A linear shape function is assumed over the boundary of the element. One
        % Gauss point is used to integrate it. Let xi denote the 1D coordinate

        ngp = 1;
        [xi,wgp] = lgwt(ngp,0,1);
        Nbmat = [1-xi 0    xi 0;
            0    1-xi  0   xi];
        q = [0 q0]';

        fe = [0; 0; Nbmat'*q*wgp*jL2]; % the first two zeros  are presnt since it referes only to side 2-3 of the
        %     trianle
        %     le = wgp*jL2;
        %     fe = jL2/2*[0 0 q(1) q(2) q(1) q(2)]'; % this is the simplest case of constant shape functon over the element boundary
        for i = 1:3
            I = conneT(ie,i);
            F(2*(I-1)+1:2*(I-1)+2,1) = F(2*(I-1)+1:2*(I-1)+2,1) +...
                fe(2*(i-1)+1:2*(i-1)+2,1);
             F2(2*(I-1)+1:2*(I-1)+2,1) = F2(2*(I-1)+1:2*(I-1)+2,1) +...
                fe(2*(i-1)+1:2*(i-1)+2,1);
        end
    end
%}

%     if issymmetric(K)~=1
%         return
%     end

%% Solve system
Z = KK\F;

%% Post process
% Deformed coordinates

Xdisp = Z(1:2:length(Z)-1);
Ydisp = Z(2:2:length(Z));
deformation = [Xdisp,Ydisp];
%     coord_def = coord(:,2:3) + deformation;
%     hdef = trimesh(conne,coord_def(:,1),coord_def(:,2));
%     axis equal


% Reaction forces
RBottom = KKold(dofB,:)*Z;
RLeftx   = KKold(dofLx,:)*Z;
% Internal stress, total and viscous deformation
for ie = 1:size(conne,1)
    vrtx = conne(ie,1:3); % vertexes array
    X = [coord(vrtx(1),2),coord(vrtx(2),2),coord(vrtx(3),2)]';
    Y = [coord(vrtx(1),3),coord(vrtx(2),3),coord(vrtx(3),3)]';
    N_xi  = [-1 1 0]';
    N_eta = [-1 0 1]';
    Jmat(1,1) = N_xi'*X; Jmat(1,2) = N_xi'*Y;
    Jmat(2,1) = N_eta'*X; Jmat(2,2) = N_eta'*Y;
    dofglob = reshape([2*vrtx-1; 2*vrtx],6,1);
    
    
    detJ = det(Jmat);
    Jm1 = 1/detJ*[Jmat(2,2), -Jmat(1,2); -Jmat(2,1), Jmat(1,1)];
    N1_x = Jm1(1,:)*[N_xi(1) N_eta(1)]';
    N1_y = Jm1(2,:)*[N_xi(1) N_eta(1)]';
    N2_x = Jm1(1,:)*[N_xi(2) N_eta(2)]';
    N2_y = Jm1(2,:)*[N_xi(2) N_eta(2)]';
    N3_x = Jm1(1,:)*[N_xi(3) N_eta(3)]';
    N3_y = Jm1(2,:)*[N_xi(3) N_eta(3)]';
    
    B = [N1_x 0    N2_x 0    N3_x 0;
        0     N1_y 0    N2_y 0    N3_y;
        0     0    0    0    0    0;
        N1_y  N1_x N2_y N2_x N3_y N3_x];
    BD = PrD*B;
    b = m'*B;
    
    vloce = Z(dofglob);
    %             sige = GDinf*B*vloce;
    %             sige_vet(ie,1:3) = sige';
    
    
    epsG(:,ie,nt) = BD*vloce;
    epsvG(1:4,1:NMeG,ie,nt) = repmat(dt./(2*tauG + dt),1,4)'.*(BD*vloce);
    tht(1,ie,nt) = b*vloce;
    thtv(1,1:NMeK,ie,nt) = dt./(2*tauK + dt)'.*(b*vloce);
    
    
    St(nt).epsG(1:4,ie)=epsG(1:4,ie,nt);    
    St(nt).tht(1,ie)=tht(1,ie,nt);
    St(nt).epsvG(1:4,1:NMeG,ie)=epsvG(1:4,1:NMeG,ie,nt);
    St(nt).thtv(1,1:NMeK,ie)=thtv(1,1:NMeK,ie,nt);
    St(nt).betGnm1=zeros(4,NMeG,ie);
    St(nt).betKnm1=zeros(1,NMeK,ie);
    % Se vuoi qui puoi calcolare lo stress \sig_D+\sig_V....
end

%% Analytical solution
%{
    %     % Benchmark pb. (see Felippa §27.3.1. p. 27-4)
    %     uy = q(2)*coord(:,3)/(GVinf + sum(GV));
    %     ux = -.2*q(2)*coord(:,2)/(GVinf + sum(GV));
    %
    %     uexact = [ux,uy];
    
    
    % % errorx = norm(deformation(:,1) - uexact(:,1))/norm(uexact(:,1))
    % % errory = norm(deformation(:,2) - uexact(:,2))/norm(uexact(:,2))
    % % sqrt(errorx^2+errory^2)
    NN
    %     % err1 = norm([deformation(:,1);deformation(:,2)] - [uexact(:,1);uexact(:,2)])/norm([uexact(:,1),uexact(:,2)])
    %     for i = 2:NN
    %         ERR(i) = norm(deformation(i,:)-uexact(i,:))/norm(uexact(i,:));
    %     end
    %     err2 = mean(ERR)/length(ERR)
    %
    % NNvec =  [12   30      120      480      1920    7680]
    % errvec = [     0.0409  0.0162   0.0071   0.0033  0.0016 ] % con err2
    % % errvec = [0.0816  0.0304   0.0125   0.0055  0.0025] % con err1
    % figure
    % loglog(NNvec,errvec)
    % hold on
    % loglog(NNvec,NNvec.^-1,'--')
    % grid on
%}
deformationx_time(:,nt) = deformation(:,1);
deformationy_time(:,nt) = deformation(:,2);
RBottomx_time(:,nt) =  RBottom(1:end/2); % RBottom è già ordinato prima le x e poi le y
RBottomy_time(:,nt) =  RBottom(end/2+1:end);
RLeftx_time(:,nt) =RLeftx;
%% Time stepping nt = 2,3...
for nt = 2:ntimestep
    t = dt*(nt-1);
    time_progress_percent = round(t/Tsim*100);
    total_simulation_time = Tsim ;
    fprintf('Time step = %1.5f of %1.5f (%1.2f%%)\n',t,Tsim,t/Tsim*100)
    KK = zeros(2*NN);
    Fvol = zeros(2*NN,1);
    Bet = zeros(2*NN,1);
    %         F   = zeros(2*NN,1);
    for ie = 1:size(conne,1)
        vrtx = conne(ie,1:3); % vertexes array
        X = [coord(vrtx(1),2),coord(vrtx(2),2),coord(vrtx(3),2)]';
        Y = [coord(vrtx(1),3),coord(vrtx(2),3),coord(vrtx(3),3)]';
        N_xi  = [-1 1 0]';
        N_eta = [-1 0 1]';
        Jmat(1,1) = N_xi'*X; Jmat(1,2) = N_xi'*Y;
        Jmat(2,1) = N_eta'*X; Jmat(2,2) = N_eta'*Y;
        
        %             betDnm1 = zeros(3,1);
        %             betVnm1 = zeros(3,1);
        
        % One Gauss point over the element
        xi_gp =  1/3; % not used actually
        eta_gp = 1/3; % not used actually
        w_gp = 0.5;
        detJ = det(Jmat);
        Ae(ie) = w_gp*detJ;
        Jm1 = 1/detJ*[Jmat(2,2), -Jmat(1,2); -Jmat(2,1), Jmat(1,1)];
        N1_x = Jm1(1,:)*[N_xi(1) N_eta(1)]';
        N1_y = Jm1(2,:)*[N_xi(1) N_eta(1)]';
        N2_x = Jm1(1,:)*[N_xi(2) N_eta(2)]';
        N2_y = Jm1(2,:)*[N_xi(2) N_eta(2)]';
        N3_x = Jm1(1,:)*[N_xi(3) N_eta(3)]';
        N3_y = Jm1(2,:)*[N_xi(3) N_eta(3)]';
        
        B = [N1_x 0    N2_x 0    N3_x 0;
            0     N1_y 0    N2_y 0    N3_y;
            0     0    0    0    0    0;
            N1_y  N1_x N2_y N2_x N3_y N3_x];
        BD = PrD*B;
        b = m'*B;
        % Element matrix (no summation since yhere is only 1 GP)
        
        keG = B'*( Ginf + sum(G) - G'*(dt./(2*tauG + dt)) )*Dmu*BD*w_gp*detJ;
        keK = b'*( Kinf + sum(K) - K'*(dt./(2*tauK + dt)) )*b*w_gp*detJ;
        ke = keG + keK;
        %
        %
        %             ke = ( B'*( Ginf + sum(G) - G'*(dt./(2*tauG + dt)) )*BD + ...
        %                 b'*( Kinf + sum(K) - K'*(dt./(2*tauK + dt)) )*b )*w_gp*detJ;
        
        epsGnm1 = epsG(1:4,ie,nt-1);
        epsvGnm1 = epsvG(1:4,1:NMeG,ie,nt-1);
        
        thtnm1 = tht(1,ie,nt-1);
        thtvnm1 = thtv(1,1:NMeK,ie,nt-1);
        
        betGnm1(1:4,1:NMeG,ie) = repmat(dt./(2*tauG + dt),1,4)'.*(epsGnm1 - epsvGnm1) + repmat(2*tauG./(2*tauG + dt),1,4)'.*epsvGnm1;
        betKnm1(1,1:NMeK,ie)   = (dt./(2*tauK + dt))'.*(thtnm1 - thtvnm1) + (2*tauK./(2*tauK + dt))'.*thtvnm1;
                
        febetnm1G = B'*Dmu*sum( repmat(G,1,4)'.*betGnm1(1:4,1:NMeG,ie),2 )*w_gp*detJ;
        febetnm1K = b'* ( K'*betKnm1(1,1:NMeK,ie)' )*w_gp*detJ;
        
        febetnm1 = febetnm1G + febetnm1K;
        
        bvol = zeros(6,1);
        St(nt).betGnm1=betGnm1;
        St(nt).betKnm1=betKnm1;
    
        % Assembly
        for i = 1:3
            I = conne(ie,i);
            for j = 1:3
                J = conne(ie,j);
                KK(2*(I-1)+1:2*(I-1)+2,2*(J-1)+1:2*(J-1)+2) = KK(2*(I-1)+1:2*(I-1)+2,2*(J-1)+1:2*(J-1)+2) +...
                    ke(2*(i-1)+1:2*(i-1)+2,2*(j-1)+1:2*(j-1)+2);
            end
            Bet(2*(I-1)+1:2*(I-1)+2,1) = Bet(2*(I-1)+1:2*(I-1)+2,1) +...
                febetnm1(2*(i-1)+1:2*(i-1)+2,1);
            Fvol(2*(I-1)+1:2*(I-1)+2,1) = Fvol(2*(I-1)+1:2*(I-1)+2,1) +...
                bvol(2*(i-1)+1:2*(i-1)+2,1);
        end
    end
    
    KKold = KK;
    F = Fvol+Bet;
    
    % Boundary conditions: for the benchmark pb. (see Felippa §27.3.1. p. 27-4) and other simple cases
    %% Bottom elements at y = 0;
    kk = 1;
i = 1;
for j = 1:nx
    ndof(kk) = (i-1)*nx + j;
    kk = kk + 1;
end
dof_bottomX = 2*(ndof-1)+1;
dof_bottomY = 2*ndof;
dofB = [dof_bottomX,dof_bottomY];
% Clamped case
for k = 1:length(dof_bottomY)
    KK(dof_bottomY(k),:) = zeros(1,2*NN);
    KK(dof_bottomY(k),dof_bottomY(k)) = 1;
end
F(dof_bottomY) = 0;
for k = ceil(nx/2)
    KK(dof_bottomX(k),:) = zeros(1,2*NN);
    KK(dof_bottomX(k),dof_bottomX(k)) = 1;
end
F(dof_bottomX) = 0;
clear ndof

    %% Lateral elements at x = 0 = symmetry axis!
    kk = 1;
    for i = 1:ny
        j = 1;
        ndof(kk) = (i-1)*nx + j;
        kk = kk + 1;
    end
    dofLx = 2*(ndof-1)+1;
    % for k = 1:length(dofLx)
    %     KK(dofLx(k),:) = zeros(1,2*NN);
    %     KK(dofLx(k),dofLx(k)) = 1;
    % end
    % F(dofLx) = 0;

% dof_leftY=dofLx(2:end);

    clear ndof
    %% Lateral elements at x = Lx;
    %{
    kk = 1;
    for i = 1:ny
        j = nx;
        ndof(kk) = (i-1)*nx + j;
        kk = kk + 1;
    end
    dofoffRx = 2*ndof-1;
    for k = 1:length(dofoffRx)
        K(dofoffRx(k),:) = zeros(1,2*NN);
        K(dofoffRx(k),dofoffRx(k)) = 1;
    end
    clear ndof
    %}
    %% Top elements at y = Ly;
    % Unit displacement at beam top
%     kk = 1;
%     i = ny;
%     for j = 1:nx
%         ndof(kk) = (i-1)*nx + j;
%         kk = kk+1;
%     end
%     %     dofTx = 2*(ndof-1)+1;
%     dofTy = 2*ndof;
%     % Unit displacement enforced
%     for k = 1:length(dofTy)
%         KK(dofTy(k),:) = zeros(1,2*NN);
%         KK(dofTy(k),dofTy(k)) = 1;
%     end
%     F(dofTy) = uc_th(nt); % Unit displacement enforced
%     clear ndof
    
    % Distributed load on top elements at y = Ly;
    %
         for ie = 1:size(conneT,1)
             vrtx = conneT(ie,1:3);
             X = [coord(vrtx(1),2),coord(vrtx(2),2),coord(vrtx(3),2)]';
             Y = [coord(vrtx(1),3),coord(vrtx(2),3),coord(vrtx(3),3)]';
             jL1 = norm([X(2) Y(2)] - [X(1) Y(1)]);
             jL2 = norm([X(3) Y(3)] - [X(2) Y(2)]);
             jL3 = norm([X(3) Y(3)] - [X(1) Y(1)]);
             dofglob = reshape([2*vrtx-1; 2*vrtx],6,1);
             dofUBx(ie,:)=dofglob(1:2:end);
             dofUBy(ie,:)=dofglob(2:2:end);
       
       % A linear shape function is assumed over the boundary of the element. One
       % Gauss point is used to integrate it. Let xi denote the 1D coordinate
        
        ngp = 1;
        [xi,wgp] = lgwt(ngp,0,1);
        Nbmat = [1-xi 0    xi 0;
            0    1-xi  0   xi];
        q = [0 q0]';
        
        fe = [0; 0; Nbmat'*q*wgp*jL2]; % the first two zeros  are presnt since it referes only to side 2-3 of the
            %trianle
            % le = wgp*jL2;
            % fe = jL2/2*[0 0 q(1) q(2) q(1) q(2)]'; % this is the simplest case of constant shape functon over the element boundary
        for i = 1:3
            I = conneT(ie,i);
            F(2*(I-1)+1:2*(I-1)+2,1) = F(2*(I-1)+1:2*(I-1)+2,1) +...
                fe(2*(i-1)+1:2*(i-1)+2,1);
              F2(2*(I-1)+1:2*(I-1)+2,1) = F2(2*(I-1)+1:2*(I-1)+2,1) +...
                fe(2*(i-1)+1:2*(i-1)+2,1);
        end
    end
    %}
    % Concntrated load at beam top
    
%         kk = 1;
%         i = ny;
%         for j = 1:nx
%             ndof(kk) = (i-1)*nx + j;
%             kk = kk+1;
%         end
%         %     dofTx = 2*(ndof-1)+1;
%         dofTy = 2*ndof;
% %             F0 = 1000;
%         F(dofTy) = F0/length(dofTy);
% 
%      clear ndof
    %% Solve system
    Z = KK\F;
    
    %% Post process
    % Deformed coordinates
    Xdisp = Z(1:2:length(Z)-1);
    Ydisp = Z(2:2:length(Z));
    deformation = [Xdisp,Ydisp];
    
    %{
        coord_def = coord(:,2:3) + deformation;
        hdef = trimesh(conne,coord_def(:,1),coord_def(:,2));
        axis equal
    %}
    
    % Reaction forces
    RBottom = KKold(dofB,:)*Z - Bet(dofB);
    RLeftx   = KKold(dofLx,:)*Z- Bet(dofLx);
    %dofLx
    % Internal stress, total and viscous deformation
    for ie = 1:size(conne,1)
        vrtx = conne(ie,1:3); % vertexes array
        X = [coord(vrtx(1),2),coord(vrtx(2),2),coord(vrtx(3),2)]';
        Y = [coord(vrtx(1),3),coord(vrtx(2),3),coord(vrtx(3),3)]';
        N_xi  = [-1 1 0]';
        N_eta = [-1 0 1]';
        Jmat(1,1) = N_xi'*X; Jmat(1,2) = N_xi'*Y;
        Jmat(2,1) = N_eta'*X; Jmat(2,2) = N_eta'*Y;
        dofglob = reshape([2*vrtx-1; 2*vrtx],6,1);
        
        
        detJ = det(Jmat);
        Jm1 = 1/detJ*[Jmat(2,2), -Jmat(1,2); -Jmat(2,1), Jmat(1,1)];
        N1_x = Jm1(1,:)*[N_xi(1) N_eta(1)]';
        N1_y = Jm1(2,:)*[N_xi(1) N_eta(1)]';
        N2_x = Jm1(1,:)*[N_xi(2) N_eta(2)]';
        N2_y = Jm1(2,:)*[N_xi(2) N_eta(2)]';
        N3_x = Jm1(1,:)*[N_xi(3) N_eta(3)]';
        N3_y = Jm1(2,:)*[N_xi(3) N_eta(3)]';
                
        B = [N1_x 0    N2_x 0    N3_x 0;
            0     N1_y 0    N2_y 0    N3_y;
            0     0    0    0    0    0;
            N1_y  N1_x N2_y N2_x N3_y N3_x];
        BD = PrD*B;
        b = m'*B;
        vloce = Z(dofglob);
        %             sige = GDinf*B*vloce;
        %             sige_vet(ie,1:3) = sige';
        
        
        epsG(1:4,ie,nt) = BD*vloce;
        epsvG(1:4,1:NMeG,ie,nt) = betGnm1(1:4,1:NMeG,ie) + repmat(dt./(2*tauG + dt),1,4)'.*epsG(1:4,ie,nt);
        
        St(nt).epsG(1:4,ie)=epsG(1:4,ie,nt);
        St(nt).epsvG(1:4,1:NMeG,ie)=epsvG(1:4,1:NMeG,ie,nt);

        tht(1,ie,nt) = b*vloce;
        thtv(1,1:NMeK,ie,nt)    = betKnm1(1,1:NMeK,ie) + dt./(2*tauK + dt)'.*tht(1,ie,nt);
        
        St(nt).tht(1,ie)=tht(1,ie,nt);
        St(nt).thtv(1,1:NMeK,ie)=thtv(1,1:NMeK,ie,nt);



        % Compute stress
        s_dev(1:4,ie,nt) = G0*Dmu*BD*vloce - Dmu*sum(repmat(G,1,4)'.*epsvG(1:4,1:NMeG,ie,nt),2);
        p(1,ie,nt)     = K0*b*vloce -   K'*thtv(1,1:NMeK,ie,nt)';
        s_tot(1:4,ie,nt) =  s_dev(1:4,ie,nt) + p(1,ie,nt)*m;
        
        eps_tot(1:4,ie,nt) = epsG(1:4,ie,nt) + tht(1,ie,nt)*m;
        
        
    end
    
    %% Analytical solution
    % Benchmark pb. (see Felippa §27.3.1. p. 27-4)
    %         uy = q(2)*coord(:,3)/GVinf;
    %         ux = -.2*q(2)*coord(:,2)/GVinf;
    %
    %         uexact = [ux,uy];
    
    
    % % errorx = norm(deformation(:,1) - uexact(:,1))/norm(uexact(:,1))
    % % errory = norm(deformation(:,2) - uexact(:,2))/norm(uexact(:,2))
    % % sqrt(errorx^2+errory^2)
    NN;
    % err1 = norm([deformation(:,1);deformation(:,2)] - [uexact(:,1);uexact(:,2)])/norm([uexact(:,1),uexact(:,2)])
    %         for i = 2:NN
    %             ERR(i) = norm(deformation(i,:)-uexact(i,:))/norm(uexact(i,:));
    %         end
    %         err2 = mean(ERR)/length(ERR)
    %
    % NNvec =  [12   30      120      480      1920    7680]
    % errvec = [     0.0409  0.0162   0.0071   0.0033  0.0016 ] % con err2
    % % errvec = [0.0816  0.0304   0.0125   0.0055  0.0025] % con err1
    % figure
    % loglog(NNvec,errvec)
    % hold on
    % loglog(NNvec,NNvec.^-1,'--')
    % grid on
    deformationx_time(:,nt) = deformation(:,1);
    deformationy_time(:,nt) = deformation(:,2);
    RBottomx_time(:,nt) =  RBottom(1:2*nx/2); % RBottom è già ordinato prima le x e poi le y
    RBottomy_time(:,nt) =  RBottom(2*nx/2+1:2*nx);
    RLeftx_time(:,nt) =   RLeftx;

end
data.deformation_time = [deformationx_time;deformationy_time];
data.stress_time = s_tot;
data.strain_time = eps_tot;
data.RBottom_time     = [RBottomx_time;RBottomy_time];
data.RBottomx_time     = RBottomx_time;
data.RBottomy_time     = RBottomy_time;
data.Tsim = Tsim;
data.dt = dt;
data.time = time;
data.dofB=dofB;

data.dof_bottomX=dof_bottomX;
data.dof_bottomY=dof_bottomY;

data.dof_leftX=dof_leftX;
data.dof_leftY=dof_leftY;

data.dof_rightX=dof_rightX(2:end-1);
data.dof_rightY=dof_rightY(2:end-1);

data.dof_topX=dof_topX(2:end);
data.dof_topY=dof_topY;

data.RLeftx_time =RLeftx_time;
data.dofUBx=dofUBx;
data.dofUBy=dofUBy;
% data.omevec = omevec;
% data.freqvec = freqvec;
% data.indvec = indvec; 
% data.fmin = fmin; 
% data.fmax = fmax; 
% data.Amp = Amp;
data.uc_th = uc_th; 


Force(1,:)=20*q0*ones(size(time))/1000;
Force(2,:)=sum(RBottomy_time,1)/1000;
Force(3:4,:)=zeros(2,length(time));


% data.T = T;
% clear deformationx_time deformationy_time s_tot eps_tot RBottomx_time RBottomy_time time
% end
filename=sprintf('Id_TD_q%1.0f_EL%2.0f_%1.0fME_dt%1.3f_Tsim%4.3f.mat',q0,size(conne,1),length(G),dt,Tsim);
save(filename,'data','St','conne','q0','coord','target_mat');

U=data.deformation_time;
% Abaqus_results
save ('data','data','-v7.3');
save ('target_mat','target_mat', '-v7.3');
% save ('workspace','-v7.3');

% %% Post-process for each omega
% for iome = 1:length(omevec)
%     deformation_time = data(iome).deformation_time;
%     deformationx_time = deformation_time(1:NN,:);
%     deformationy_time = deformation_time(NN+1:2*NN,:);
%     time = data(iome).time;
%     
%     figure
%     hold on
%     plot(time,deformationx_time(NN,:))
%     plot(time,deformationy_time(NN,:))
%     grid on
%     legend('x direction','y direction')
%     xlabel('time')
%     ylabel('Displacements')
%     hold off
% end

figure
hold on
h = trimesh(conne,coord(:,2),coord(:,3),zeros(NN,1),'LineStyle','-','edgecolor',[.5 .5 ,.5],'FaceAlpha',0);
coord_def = coord(:,2:3) + [deformationx_time(:,end),deformationy_time(:,end)];

for i=1:size(coord,1)
  text(coord(i,2),coord(i,3),num2str(2*i));
  hold on
end
hdef = trimesh(conne,coord_def(:,1),coord_def(:,2),zeros(NN,1),'LineStyle','-','edgecolor',[0 0 0],'FaceAlpha',0);
view([0 0 1])
grid on
axis equal

%% Post-process and comparisons with Abaqus single omega
%{



  % Compare with Abaqus
    figure
    hold on
    title(sprintf('node = %i',1))
    plot(time,RBottomx_time(1,:),'r-')
    plot(Rx_n1(:,1),Rx_n1(:,2),'r--')
    plot(time,RBottomy_time(1,:),'b-')
    plot(Ry_n1(:,1),Ry_n1(:,2),'b--')
    legend('Rx my FEM','Rx Abaqus','Ry my FEM','Ry Abaqus')
    xlabel('time')
    ylabel('Reaction forces')
    grid on
    
    figure
    hold on
    title(sprintf('node = %i',6))
    plot(time,RBottomx_time(6,:),'r-')
    plot(Rx_n6(:,1),Rx_n6(:,2),'r--')
    plot(time,RBottomy_time(6,:),'b-')
    plot(Ry_n6(:,1),Ry_n6(:,2),'b--')
    legend('Ry my FEM','Ry Abaqus','Ry my FEM','Ry Abaqus')
    xlabel('time')
    ylabel('Reaction forces')
    grid on






% Plot initial and def mesh
figure
hold on
h = trimesh(conne,coord(:,2),coord(:,3),zeros(NN,1),'LineStyle','-','edgecolor',[.5 .5 ,.5],'FaceAlpha',0);
coord_def = coord(:,2:3) + [deformationx_time(:,end),deformationy_time(:,end)];
hdef = trimesh(conne,coord_def(:,1),coord_def(:,2),zeros(NN,1),'LineStyle','-','edgecolor',[0 0 0],'FaceAlpha',0);
view([0 0 1])
grid on
axis equal
      

figure
hold on
plot(time,deformationx_time(end-nx+1:end,:))
xlabel('time')
ylabel('Tip displacements in x')
grid on

figure
hold on
plot(time,deformationy_time(end-nx+1:end,:))
xlabel('time')
ylabel('Tip displacements in y')
grid on

% Bottom-left (mid node: sull'asse di simmetria)
figure
hold on
plot(time,RBottomx_time(1,:))
plot(Rx_nmid(:,1),Rx_nmid(:,2))
legend('my FEM', 'Abaqus')
xlabel('time')
ylabel('Reaction in x ')
grid on

figure
hold on
plot(time,RBottomy_time(1,:))
plot(Ry_nmid(:,1),Ry_nmid(:,2))
legend('my FEM', 'Abaqus')
xlabel('time')
ylabel('Reaction in y ')
grid on

% Bottom-left+1 (primo nodo interno da estremo bottom left)
figure
hold on
plot(time,RBottomx_time(2,:))
plot(Rx_n2(:,1),Rx_n2(:,2))
legend('my FEM', 'Abaqus')
xlabel('time')
ylabel('Reaction in x ')
grid on

figure
hold on
plot(time,RBottomy_time(2,:))
plot(Ry_n2(:,1),Ry_n2(:,2))
legend('my FEM', 'Abaqus')
xlabel('time')
ylabel('Reaction in y ')
grid on

% Bottom-right node
figure
hold on
plot(time,RBottomx_time(end,:))
plot(Rx_nbr(:,1),Rx_nbr(:,2))
legend('my FEM', 'Abaqus')
xlabel('time')
ylabel('Reaction in x ')
grid on

figure
hold on
plot(time,RBottomy_time(end,:))
plot(Ry_nbr(:,1),Ry_nbr(:,2))
legend('my FEM', 'Abaqus')
xlabel('time')
ylabel('Reaction in y ')
grid on

% % %% Bottom-right-1 node (secondo nodo dall'esterno)
% % figure
% % hold on
% % plot(time,RBottomx_time(end-1,:))
% % plot(Rx_n2(:,1),Rx_n2(:,2))
% % legend('my FEM', 'Abaqus')
% % xlabel('time')
% % ylabel('Reaction in x ')
% % grid on
% %
% % figure
% % hold on
% % plot(time,RBottomy_time(end-1,:))
% % plot(Ry_n2(:,1),Ry_n2(:,2))
% % legend('my FEM', 'Abaqus')
% % xlabel('time')
% % ylabel('Reaction in y ')
% % grid on
% %
% % %% Bottom-right-2 node (terzo nodo dall'esterno)
% % figure
% % hold on
% % plot(time,RBottomx_time(3,:))
% % plot(Rx_n3(:,1),Rx_n3(:,2))
% % legend('my FEM', 'Abaqus')
% % xlabel('time')
% % ylabel('Reaction in x ')
% % grid on
% %
% % figure
% % hold on
% % plot(time,RBottomy_time(3,:))
% % plot(Ry_n3(:,1),Ry_n3(:,2))
% % legend('my FEM', 'Abaqus')
% % xlabel('time')
% % ylabel('Reaction in y ')
% % grid on
%
%
% Internal displacements node 10
figure
hold on
plot(time,deformationx_time(10,:),'r-')
plot(Ux_n10(:,1),Ux_n10(:,2),'r--')
plot(time,deformationy_time(10,:),'b-')
plot(Uy_n10(:,1),Uy_n10(:,2),'b--')
legend('Ux my FEM','Ux Abaqus','Uy my FEM','Uy Abaqus')
xlabel('time')
ylabel('Displacements')
grid on
% ylim([0 4e-3])
%
% figure
% hold on
% plot(time,deformationx_time(66,:))
% plot(Ux_n66(:,1),Ux_n66(:,2))
% plot(time,deformationy_time(66,:))
% plot(Uy_n66(:,1),Uy_n66(:,2))
% legend('Ux my FEM','Ux Abaqus','Uy my FEM','Uy Abaqus')
% xlabel('time')
% ylabel('Displacements')
% grid on
% % ylim([0 6e-3])
figure
hold on
plot(time,deformationx_time(306,:),'r-')
plot(Ux_ntr(:,1),Ux_ntr(:,2),'r--')
plot(time,deformationy_time(306,:),'b-')
plot(Uy_ntr(:,1),Uy_ntr(:,2),'b--')
legend('Ux my FEM','Ux Abaqus','Uy my FEM','Uy Abaqus')
xlabel('time')
ylabel('Displacements')
grid on

figure
hold on
plot(time,deformationx_time(302,:),'r-')
plot(Ux_n302(:,1),Ux_n302(:,2),'r--')
plot(time,deformationy_time(302,:),'b-')
plot(Uy_n302(:,1),Uy_n302(:,2),'b--')
legend('Ux my FEM','Ux Abaqus','Uy my FEM','Uy Abaqus')
xlabel('time')
ylabel('Displacements')
grid on


% Internal stress at Gauss (single one) point of element 30
figure
hold on
plot(time,squeeze(s_tot(1,15,:)),'r-')
plot(s11_e_15_GP(:,1),s11_e_15_GP(:,2),'r--')

plot(time,squeeze(s_tot(2,15,:)),'b-')
plot(s22_e_15_GP(:,1),s22_e_15_GP(:,2),'b--')

plot(time,squeeze(s_tot(3,15,:)),'g-')
plot(s33_e_15_GP(:,1),s33_e_15_GP(:,2),'g--')

plot(time,squeeze(s_tot(4,15,:)),'m-')
plot(s12_e_15_GP(:,1),s12_e_15_GP(:,2),'m--')

legend('\sigma_{11} my FEM','\sigma_{11} Abaqus','\sigma_{22} my FEM','\sigma_{22} Abaqus',...
       '\sigma_{33} my FEM','\sigma_{33} Abaqus','\sigma_{12} my FEM','\sigma_{12} Abaqus')
xlabel('time')
ylabel('Stress')
grid on

figure
hold on
plot(time,squeeze(s_tot(1,247,:)),'r-')
plot(s11_e_247_GP(:,1),s11_e_247_GP(:,2),'r--')

plot(time,squeeze(s_tot(2,247,:)),'b-')
plot(s22_e_247_GP(:,1),s22_e_247_GP(:,2),'b--')

plot(time,squeeze(s_tot(3,247,:)),'g-')
plot(s33_e_247_GP(:,1),s33_e_247_GP(:,2),'g--')

plot(time,squeeze(s_tot(4,247,:)),'m-')
plot(s12_e_247_GP(:,1),s12_e_247_GP(:,2),'m--')

legend('\sigma_{11} my FEM','\sigma_{11} Abaqus','\sigma_{22} my FEM','\sigma_{22} Abaqus',...
       '\sigma_{33} my FEM','\sigma_{33} Abaqus','\sigma_{12} my FEM','\sigma_{12} Abaqus')
xlabel('time')
ylabel('Stress')
grid on
% ylim([0 4e-3])
%}
