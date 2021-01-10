int: r; % Number of rounds


array[0..r, 0..15] of var 0..1: A; 
array[0..r, 0..15] of var 0..1: B; 

array[0..r-1, 0..15] of var 0..1: U; 
array[0..r-1, 0..15] of var 0..1: V; 
array[0..r-1, 0..15] of var 0..1: W;
array[0..r-1, 0..15] of var 0..1: T;


constraint forall (n in 0..r-1, j in 0..15)(A[n,j]=U[n,j]+V[n,j]+W[n,j]+B[n+1,j]);

constraint forall (n in 0..r-1, j in 0..15)(T[n,j]>=U[n,(j+5) mod 16]);

constraint forall (n in 0..r-1, j in 0..15)(T[n,j]>=V[n,j mod 16]);


constraint forall (n in 0..r-1, j in 0..15)(T[n,j]<=U[n,(j+1) mod 16]+V[n,j mod 16]);

constraint forall (n in 0..r-1, j in 0..15)(A[n+1,j]=B[n,j]+T[n,j]+W[n,(j+1) mod 16]);


constraint sum(j in 0..15)(A[0,j]+B[0,j])=4;
constraint sum(j in 0..3)(B[0,j])=4;

constraint sum(j in 0..15)(A[r,j]+B[r,j])=1;
constraint B[r,10]=1;

solve satisfy;
