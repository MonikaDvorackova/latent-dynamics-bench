import math,random
def solve(a,b):
    n=len(b);m=[list(map(float,a[i]))+[float(b[i])] for i in range(n)]
    for c in range(n):
        p=max(range(c,n),key=lambda r:abs(m[r][c]));m[c],m[p]=m[p],m[c]
        if abs(m[c][c])<1e-12:raise ValueError('singular design')
        q=m[c][c];m[c]=[v/q for v in m[c]]
        for r in range(n):
            if r!=c:q=m[r][c];m[r]=[x-q*y for x,y in zip(m[r],m[c])]
    return [m[i][-1] for i in range(n)]
def fit(xs,us,ys):
    # ridge-stabilized normal equations for x_next = a*x + b*u + c
    z=[[x,u,1.0] for x,u in zip(xs,us)];d=3;mat=[[sum(row[i]*row[j] for row in z)+(1e-6 if i==j else 0) for j in range(d)] for i in range(d)];rhs=[sum(row[i]*y for row,y in zip(z,ys)) for i in range(d)];return solve(mat,rhs)
def main():
    r=random.Random(9);x=0.;xs=[];us=[];ys=[]
    for _ in range(1000):u=r.uniform(-1,1);xn=.92*x+.35*u+.01*r.gauss(0,1);xs.append(x);us.append(u);ys.append(xn);x=xn
    a,b,c=fit(xs,us,ys);truth=pred=0.;one=[];roll=[]
    for _ in range(300):u=r.uniform(-1,1);truth=.92*truth+.35*u;pred=a*pred+b*u+c;roll.append((truth-pred)**2)
    print({'learned_transition':[a,b,c],'rollout_rmse':math.sqrt(sum(roll)/len(roll))})
if __name__=='__main__':main()
