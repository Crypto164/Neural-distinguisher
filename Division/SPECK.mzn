int: r; % Number of rounds


array[0..r, 0..15] of var 0..1: X; 
array[0..r-1, 0..15] of var 0..1: XR; 
array[0..r, 0..15] of var 0..1: Y;

array[0..r-1, 0..15] of var 0..1: YA;
array[0..r-1, 0..15] of var 0..1: YB;
array[0..r-1, 0..15] of var 0..1: Z;
array[0..r-1, 0..15] of var 0..1: ZA;

array[0..r-1, 0..3,0..12] of var 0..1: U;
array[0..r-1, 0..3,0..2] of var 0..1: V;
array[0..r-1, 0..3,0..1] of var 0..1: M;
array[0..r-1, 0..3,0..3] of var 0..1: T;
array[0..r-1, 0..3,0..1] of var 0..1: R;
array[0..r-1, 0..3,0..1] of var 0..1: G;
array[0..r-1, 0..3,0..1] of var 0..1: Q;
array[0..r-1, 0..3,0..1] of var 0..1: W;

constraint forall (j in 0..3)(X[0,j]=1);
constraint forall (j in 4..15)(X[0,j]=0);
constraint forall (j in 0..15)(Y[0,j]=0);


%Rotation-COPY
constraint forall (n in 0..r-1, j in 0..15)(Y[n,j]=YA[n,j]+YB[n,j]);
constraint forall (n in 0..r-1, j in 0..15)(X[n,j]=XR[n,(j+7) mod 16]);
constraint forall (n in 0..r-1, j in 0..15)(Z[n,j]=ZA[n,j]+X[n+1,j]);

%Xor
constraint forall (n in 0..r-1, j in 0..15)(Y[n+1,j]=ZA[n,j]+YB[n,(j+2) mod 16]);



%Add
constraint forall (n in 0..r-1, j in 0..3)(XR[n,(j*4)+3]=T[n,j,0]+T[n,j,1]);
constraint forall (n in 0..r-1, j in 0..3)(YA[n,(j*4)+3]=T[n,j,2]+T[n,j,3]);
constraint forall (n in 0..r-1, j in 0..3)(XR[n,(j*4)+2]=U[n,j,0]+U[n,j,1]+U[n,j,2]);
constraint forall (n in 0..r-1, j in 0..3)(XR[n,(j*4)+1]=U[n,j,3]+U[n,j,4]+U[n,j,5]);
constraint forall (n in 0..r-1, j in 0..3)(YA[n,(j*4)+2]=U[n,j,6]+U[n,j,7]+U[n,j,8]);
constraint forall (n in 0..r-1, j in 0..3)(YA[n,(j*4)+1]=U[n,j,9]+U[n,j,10]+U[n,j,11]);

constraint forall (n in 0..r-1, j in 0..3)(Z[n,(j*4)+3]=T[n,j,0]+T[n,j,2]);
constraint forall (n in 0..r-1, j in 0..3)(V[n,j,0]>=T[n,j,1]);
constraint forall (n in 0..r-1, j in 0..3)(V[n,j,0]>=T[n,j,3]);

constraint forall (n in 0..r-1, j in 0..3)(V[n,j,0]<=T[n,j,1]+T[n,j,3]);
constraint forall (n in 0..r-1, j in 0..3)(V[n,j,1]>=U[n,j,1]);
constraint forall (n in 0..r-1, j in 0..3)(V[n,j,1]>=U[n,j,7]);

constraint forall (n in 0..r-1, j in 0..3)(V[n,j,1]<=U[n,j,1]+U[n,j,7]);
constraint forall (n in 0..r-1, j in 0..3)(V[n,j,2]>=U[n,j,4]);
constraint forall (n in 0..r-1, j in 0..3)(V[n,j,2]>=U[n,j,10]);

constraint forall (n in 0..r-1, j in 0..3)(V[n,j,2]<=U[n,j,4]+U[n,j,10]);
constraint forall (n in 0..r-1, j in 0..3)(U[n,j,2]+U[n,j,8]=M[n,j,0]);
constraint forall (n in 0..r-1, j in 0..3)(U[n,j,5]+U[n,j,11]=M[n,j,1]);
constraint forall (n in 0..r-1, j in 0..3)(G[n,j,0]+R[n,j,0]=V[n,j,0]);

constraint forall (n in 0..r-1, j in 0..3)(Z[n,(j*4)+2]=U[n,j,0]+U[n,j,6]+G[n,j,0]);
constraint forall (n in 0..r-1, j in 0..3)(Q[n,j,0]>=M[n,j,0]);
constraint forall (n in 0..r-1, j in 0..3)(Q[n,j,0]>=R[n,j,0]);
constraint forall (n in 0..r-1, j in 0..3)(Q[n,j,0]<=M[n,j,0]+R[n,j,0]);

constraint forall (n in 0..r-1, j in 0..3)(V[n,j,1]+Q[n,j,0]=W[n,j,0]);
constraint forall (n in 0..r-1, j in 0..3)(G[n,j,1]+R[n,j,1]=W[n,j,0]);
constraint forall (n in 0..r-1, j in 0..3)(Z[n,(j*4)+1]=U[n,j,3]+U[n,j,9]+G[n,j,1]);
constraint forall (n in 0..r-1, j in 0..3)(Q[n,j,1]>=M[n,j,1]);
constraint forall (n in 0..r-1, j in 0..3)(Q[n,j,1]>=R[n,j,1]);
constraint forall (n in 0..r-1, j in 0..3)(Q[n,j,1]<=M[n,j,1]+R[n,j,1]);
constraint forall (n in 0..r-1, j in 0..3)(V[n,j,2]+Q[n,j,1]=W[n,j,1]);
constraint forall (n in 0..r-1, j in 0..3)(X[n,(j*4)]+YA[n,(j*4)]+W[n,j,1]=Z[n,(j*4)]);



constraint sum(j in 0..15)(X[r,j]+Y[r,j])=1;
constraint Y[r,0]=1;

solve satisfy;
