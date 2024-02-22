#!/usr/bin/env python
# coding: utf-8

# In[40]:


import numpy as np
import math
import matplotlib.pyplot as plt


# # Finding roots of non-linear equations

# 1. Bisection

# In[16]:


def bracketing(f1,a,b):
    if abs(f1(a))<abs(f1(b)):
        a=a-0.5*(b-a)
    if abs(f1(a))>abs(f1(b)):
        b=b+0.5*(b-a)
    return a,b


def bisection(f,a0,b0):
    n=0
    root=[]
    e=0.000001
    while abs(b0-a0)>e and (f(a0)>e or f(b0)>e):
        c0=(a0+b0)/2
        if f(c0)*f(a0)<0:
            b0=c0
        elif f(c0)*f(b0)<0:
            a0=c0
        n+=1
        root.append((a0+b0)/2)
    return root,n


# 2. Regula Falsi

# In[17]:


def RegulaFalsi(f,a0,b0):
    c0=0
    n2=0
    c=[]
    e=0.000001
    while abs(b0-a0)>e and f(a0)>e or f(b0)>e:
        c_prev=c0
        c0=b0-(b0-a0)*f(b0)/(f(b0)-f(a0))
        if f(a0)*f(c0)<0:
            b0=c0
            if abs(c_prev-c0)<e:
                break
        if f(b0)*f(c0)<0:
            a0=c0
            if abs(c_prev-c0)<e:
                break
        n2+=1
        c.append(c0)
    return c,n2


# 3. Newton Raphson

# In[18]:


def df(f,x):
    h=0.001
    return (f(x+h)-f(x-h))/(2*h)

def NewtonRaphson(f,x0,x):
    e=0.000001
    x_ar = []
    while abs(x-x0)>e or f(x)>e:
        x0=x
        x=x0-f(x0)/df(f,x0)
        x_ar.append(x)
    return x_ar


# 4. Fixed Point

# In[19]:


#Fixed Point for single variable
def FixedPoint_single(x_k):
    x_k1, e = 1, 0.00001
    g = x_k
    while abs(x_k1 - x_k)/abs(x_k1)>e:
        g = f1(x_k1) + x_k1
        x_k, x_k1 = x_k1, g
        #print(g)
    return g


# In[41]:


#Fixed Point for multi-variable
def g1(varlist):
    return (10 - varlist[0]*varlist[1])**(1/2)
def g2(varlist):
    return ((57-varlist[1])/(3*varlist[0]))**(1/2)

def FixedPoint_multi(glist,x0list,tol):

    if len(glist)!=len(x0list):
        raise IndexError("The number of functions and initial guesses are not equal")
    else:
        for i in range(len(glist)):
            x0list[i] = (glist[i](x0list))
        step=1
        flag=1
        while flag==1:
            if step>100:
                print("The roots are not converging")
                return x0list,step
            else:
                temp = x0list[:]

                for i in range(len(glist)):
                    x0list[i] = (glist[i](x0list))
                step+=1

            for j in range(len(x0list)):
                if abs(temp[j] - x0list[j]) / x0list[j] < tol:
                    flag = 0
                else:
                    flag = 1
                    break
        return x0list,step


# 5. Newton Raphson for Multivariables

# In[29]:


def newton_raphson_multivariable(f, J, x0, tol):
    """This function is used to find the root of a given function using Newton-Raphson method.
    
    Parameters:
    - f: The system of functions. f 
    - J: The Jacobian matrix of f.
    - x0: The initial guess.
    - tol: The tolerance.
    
    Returns:
    - x: The root of the system of functions.
    - iter: The number of iterations.
    """
    x = x0
    iter = 0
    while True:
        x = x - np.matmul(np.linalg.inv(J(x)), f(x))
        iter += 1
        if np.linalg.norm(f(x)) < tol:
            break
    return x, iter

def jacobian(f, x, h=1e-5):
    """This function is used to find the Jacobian matrix of a system of functions.
    
    Parameters:
    - f: The system of functions.
    - x: The point at which the Jacobian matrix is to be evaluated.
    - h: The step size.
    
    Returns:
    - J: The Jacobian matrix of f.
    """
    n = len(f(x))
    m = len(x)
    J = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            x1 = x.copy()
            x2 = x.copy()
            x1[j] += h
            x2[j] -= h
            J[i, j] = (f(x1)[i] - f(x2)[i]) / (2 * h)
    return J

# Example usage:
# f = lambda x: np.array([x[0]**2 + x[1]**2 - 1, x[0] - x[1]])
# J = lambda x: np.array([[2*x[0], 2*x[1]], [1, -1]])
# x0 = np.array([0.5, 0.5])
# tol = 1e-5
# root, iter = newton_raphson_multivariable(f, J, x0, tol)
# solution should be [0.70710678, 0.70710678]
# print(root, iter)


# # Numerical Integration

# 1. Midpoint method

# In[21]:


def midmethod(N,b,a,f):
    h=(b-a)/N
    x_i=[0 for i in range(N)]
    for i in range(N):
        x_i[i]=((a+h*i)+(a+h*(i+1)))/2
    sum=0
    for i in range(N):
        sum=sum+h*f((x_i[i]))
    return sum


# 2. Trapezoidal method

# In[22]:


def trapezmethod(N,b,a,f):
    h=(b-a)/N
    x_i=[0 for i in range(N+1)]
    for i in range(N+1):
        x_i[i]=a+i*h
    sum=0
    for i in range(N+1):
        if i==0 or i==N:
            sum = sum + 0.5*h*f((x_i[i]))
        else:
            sum = sum+ h*f((x_i[i]))
    return sum


# 3. Simpson's Rule

# In[23]:


def SimpMeth(N,b,a,f):
    interval = (b-a)/N
    x_i=[0 for i in range(N+1)]
    
    for i in range(0,N+1,2):
        x_i[i]=a+i*interval
        
    for i in range(1,N,2):
        x_i[i]=(x_i[i-1] + x_i[i+1])/2
    
    h=(x_i[2]-x_i[0])/2
    sum=0
    for i in range(N+1):
        if i==0 or i==N:
            sum = sum + (h*f((x_i[i])))/3
        elif i%2==0:
            sum = sum + (2*h*f((x_i[i])))/3
        else:
            sum = sum + (4*h*f((x_i[i])))/3
    return sum


# 4. Gaussian Quadrature

# In[44]:


#Derivative of Legendre Polynom
def Pn_dr(x,n):
    if n == 0:
        return 0
    elif n == 1:
            return 1
    else:
        return (n*(Pn(x,n-1)-x*Pn(x,n)))/(1-x**2)

#Newton-Raphson for finding roots of legendre polynomial
def NewtonRaphson_leg(f,x0,num,x):
    e=0.000001
    while abs(x-x0)>e or f(x,num)>e:
        x0=x
        x=x0-f(x0,num)/Pn_dr(x0,num)
    return x


# In[25]:


#Gaussian Quadrature Method
def Pn(x,n): #defining Legendre Polynomial
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return ((2*n-1)*x*Pn(x,n-1)-(n-1)*Pn(x,n-2))/n   
    
#finds roots of Legendre Polynomial
def Pn_roots(num):
    roots = []
    for i in range(1, num + 1):
        guess = np.cos((2*i - 1) * np.pi / (2 * num))
        guess1 = guess + 2
        result = NewtonRaphson(Pn, guess, num, guess1)
        if result != 0:
            roots.append(result)
    return roots

#finds weight
def weight(n):
    roots=Pn_roots(n)
    weights=[]
    for i in range(n):
        w=2/((1-roots[i]**2)*(Pn_dr(roots[i],n))**2)
        weights.append(w)
    return weights

def GaussianQuad(deg,b,a):
    sum = 0
    weights=weight(deg)
    roots=Pn_roots(deg)
    for i in range(deg):
        y=(((b-a)*roots[i])/2)+((b+a)/2)
        wf=weights[i]*f2(y)
        sum+=wf
    val=((b-a)*sum)/2
    return val


# # Diff eqn - ODE

# 1. Semi implicit euler

# In[36]:


def semi_implicit(x,v,dt, t_max):
    t = 0.0
    while t<=t_max:
        v = v + dt*g(x,t)
        x = x + dt*f(v,t)
        t += dt
    return v,x


# 2. Verlet method

# In[38]:


#Verlet
def verlet(x, a, dt):
    prev_x = x
    time = 0.0

    while (x > 0):
        time += dt
        temp_x = x
        x = x * 2 - prev_x + a * dt * dt
        vel = (prev_x + temp_x)/(2*dt)
        prev_x = temp_x
    return time, vel


# 3. Velocity Verlet

# In[39]:


def velocity_verlet(x, a, dt):
    prev_x = x
    time = 0.0
    vel = 0.0
    while (x > 0.0):
        time += dt
        x += vel * dt + 0.5 * a * dt * dt;
        vel += a * dt;
    return time, vel


# 4. Leapfrog

# In[ ]:


#not complete
def leapfrog(F, x0, p0, dt, num_steps):
   
    x_values = np.zeros(num_steps + 2)
    p_values = np.zeros(num_steps + 2)
    t_values = np.zeros(num_steps + 2)

    x_values[0] = x0
    p_values[0] = p0
    t_values[0] = 0.0

    for i in range(num_steps):
        # Leapfrog integration steps
        p_half = p_values[i] + 0.5 * dt * F(x_values[i])
        x_values[i + 1] = x_values[i] + dt * p_half
        p_values[i + 1] = p_half + dt * F(x_values[i + 1])
        t_values[i + 1] = t_values[i] + dt
    
    #final half steps to get to end of trajectory
    p_half = p_values[i] + 0.5 * dt * F(x_values[i])
    x_values[i+1] = x_values[i] + p_half
    p_values[i+1] = p_half + 0.5*dt*dt
    

    return x_values, p_values, t_values


# 4. CoupledODE: Shooting method with RK4

# In[33]:


#RK4 for heat equation
def CoupledODE1(y0,z0,dzdx,dTdx):
    dx = 0.01
    T, z, x = [], [], []
    T.append(y0)
    z.append(z0)
    x.append(0)
    for i in range(1000):
        k1T=(dx*dTdx(z[i],x[i]))
        k1z=(dx*dzdx(T[i],z[i],x[i]))
        
        k2T=(dx*dTdx((z[i]+k1z/2),(x[i]+dx/2)))
        k2z=(dx*dzdx((T[i]+k1T/2),z[i]+k1z/2,(x[i]+dx/2)))
        
        k3T=(dx*dTdx((z[i]+k2z/2),(x[i]+dx/2)))
        k3z=(dx*dzdx((T[i]+k2T/2),z[i]+k1z/2,(x[i]+dx/2)))
        
        k4T=(dx*dTdx((z[i]+k2z),(x[i]+dx)))
        k4z=( dx*dzdx((T[i]+k2T),z[i]+k1z,(x[i]+dx)))
        
        T.append( T[i] + (k1T + 2*k2T + 2*k3T + k4T)/6)
        z.append(z[i] + (k1z + 2*k2z + 2*k3z + k4z)/6)
        x.append(x[i]+dx)
      
    return T, x

def ShootMeth(zeta_h,al,be,dzdx,dTdx):
    e = 0.0001
    T_zeta_h,x = CoupledODE1(al,zeta_h,dzdx,dTdx)
    plt.plot(x,T_zeta_h)
    zeta_l=5
    T_zeta_l,x = CoupledODE1(al,zeta_l,dzdx,dTdx)
    plt.plot(x,T_zeta_l)

    if abs(zeta_l-be)>e:
        l=len(T_zeta_l)-1
        zeta_l=zeta_l + (zeta_h - zeta_l)*(be-T_zeta_l[l])/(T_zeta_h[l]-T_zeta_l[l])
        T_zeta_h=T_zeta_l
        T_zeta_l,x = CoupledODE1(al,zeta_l,dzdx,dTdx)
    return T_zeta_l,x,zeta_l


# # Diff eqn - PDE

# 1. Finite difference methods

# In[31]:


def PDEExplicit(N,nx,nt,lx,lt):
    #N=time step
    hx, ht = lx/nx, lt/nt
    a = ht/hx**2

    V0, V1 = [0], [0]
    for i in range(nx+1):
        V1.append(0)
    for i in range(nx+1):
        if i==10:
            V0.append(300)
        else:
            V0.append(0)

    #as its a sparse matrix we don't create the whole matrix  
    for j in range(N):
        for i in range(nx+1):
            if i==0:
                V1[i]=(1-2*a)*V0[i] + (a*V0[i+1])
            elif i==(nx):
                V1[i]=(a*V0[i-1]) + (1-2*a)*V0[i]
            else:
                V1[i]=(a*V0[i-1]) + (1-2*a)*V0[i] + (a*V0[i+1])
        
        for i in range(nx+1):
            V0[i] = V1[i]
            
    if N==0:
        return V0
    else:
        return V1


# In[ ]:





# 2. Du Fort and Frankel

# In[ ]:


def du_fort_frankel_solve(g: callable,a: callable,b: callable,x_i: float,x_f: float,t_i: float,t_f: float,Nx: int, Nt: int):
    hx = (x_f - x_i) / Nx
    ht = (t_f - t_i) / Nt
    alpha = ht / (hx**2)
    x=[(x_i + i*hx) for i in range(0,Nx+1)]
    ulist = np.zeros((Nx+1,Nt+1))
    for i in range(Nx+1):
        ulist[i][0] = g(x[i])
    for i in range(Nt+1):
        ulist[0][i] = a(t_i + i*ht)
        ulist[Nx][i] = b(t_i + i*ht)

    a1 = (1 - 2*alpha)/(1 + 2*alpha)
    a2 = 2*alpha/(1 + 2*alpha)
    return ulist


# 3. Crank-Nicolson

# 4. Laplace eqn

# In[42]:


def solve_laplace(u, max_iter=1000, tol=1e-6):
  u1 = np.copy(u)
  for k in range(max_iter):
      u = np.copy(u1)
      for i in range(0, Ny-1):
          for j in range(0, Nx-1):
              u1[i, j] = (0.25) * (u[i+1, j] + u[i-1,j] + u[i,j+1] + u[i,j-1])
      if np.max(np.abs(u1 - u)) < tol:
          break
  return u1


# 5. PDE solver: explicit

# In[34]:


def PDEExplicit(N,nx,nt,lx,lt):
    #N=time step
    hx, ht = lx/nx, lt/nt
    a = ht/hx**2

    V0, V1 = [0], [0]
    for i in range(nx+1):
        V1.append(0)
    for i in range(nx+1):
        if i==10:
            V0.append(300)
        else:
            V0.append(0)

    #as its a sparse matrix we don't create the whole matrix  
    for j in range(N):
        for i in range(nx+1):
            if i==0:
                V1[i]=(1-2*a)*V0[i] + (a*V0[i+1])
            elif i==(nx):
                V1[i]=(a*V0[i-1]) + (1-2*a)*V0[i]
            else:
                V1[i]=(a*V0[i-1]) + (1-2*a)*V0[i] + (a*V0[i+1])
        
        for i in range(nx+1):
            V0[i] = V1[i]
            
    if N==0:
        return V0
    else:
        return V1


# 6. Wave equation

# In[43]:


def solve_wave_eqn(u, max_iter=1000, tol=1e-6):
    u1 = np.copy(u)
    for k in range(max_iter):
        u = np.copy(u1)
        for i in range(0, Ny-1):
            for j in range(0, Nx-1):
                u1[i, j+1] = 2*u[i,j] - u[i,j-1] + dt*dt*(u[i+1,j]+u[i-1,j]-2*u[i,j])/(dx*dx)
        if np.max(np.abs(u1 - u)) < tol:
            break
    return u1


# # Matrices Inversion

# In[6]:


#Check if matrices are symmetric to use any of the methods below
def SymmetricCheck(l):
    c=0
    for i in range(len(l)):
        for j in range(len(l)):
            if l[i][j]!=l[j][i]:
                c=c+1
    if c!=0:
        return False
    else:
        return True


# 1. Gauss-Jordan method

# In[1]:


#Solving linear equation using Gauss-Jordan
def Reduce(l,n):
    c,flag,m,t=0,0,0,0
    for i in range(n):
        if l[i][i]==0:
            c=1
            while (i+c)<n and l[i+c][i]==0:
                c=c+1
            if (i+c)==n:
                flag=1
                break
            j=i
            l[j],l[j+c]=l[j+c],l[j]
        for j in range(n):
            if i!=j:
                p=l[j][i]/l[i][i]
                k=0
                for k in range(n+1):
                    l[j][k]=l[j][k]-p*l[i][k]
    return flag

def PrintSolutions(l, n, c):
    print("Result is : ")
    if (c == 2):
        print("Infinite Solutions Exists")
    elif (c == 3):
        print("No Solution Exists")

    # Printing the solution by dividing constants by
    # their respective diagonal elements
    else:
        for i in range(n):
            print(l[i][n] / l[i][i], end=" ")


# In[2]:


#Find inverse of matrix using gauss-jordan
def InverseGJ(l):
    for i in range(len(l)):
        for j in range(len(l),2*len(l)):
            if j==(i+len(l)):
                l[i].append(1)
            else:
                l[i].append(0)
    for i in range(len(l)-1,0,-1):
        if l[i-1][0]<l[i][0]:
            l[i-1],l[i]=l[i],l[i-1]
            
    for i in range(len(l)):
        for j in range(len(l)):
            if i!=j:
                p=l[j][i]/l[i][i]
                for k in range(2*len(l)):
                    l[j][k]=l[j][k]-p*l[i][k]
    for i in range(len(l)):
        p=l[i][i]
        for j in range(2*len(l)):
            l[i][j]=l[i][j]/p

    l_inv=[[0 for col in range(len(l))] for row in range(len(l))]
    for i in range(len(l)):
        for j in range(len(l),2*len(l)):
            l_inv[i][j-len(l)]=l[i][j]
    return l_inv


# 2. LU Decomposition

# In[3]:


#returns lower and upper matrices
def UandL(A):
    u=[[0 for col in range(len(A))] for row in range(len(A))]
    l=[[0 for col in range(len(A))] for row in range(len(A))]
    for i in range(len(A)):
        u[0][i]=A[0][i]
        l[i][i]=1
    for j in range(len(A)):
        for i in range(1,j+1):
            sum=0
            for k in range(i):
                sum=sum+(l[i][k])*(u[k][j])
            u[i][j]=A[i][j]-sum

        for i in range(j,len(A)):
            sum=0
            for k in range(j):
                sum=sum+(l[i][k])*(u[k][j])
            if u[j][j]==0:
                continue
            else:
                l[i][j]=(A[i][j]-sum)/(u[j][j])
    return u,l


# In[4]:


#Find solutions of set of linear equations
def LUForBack(l,U,L):
    y=[0 for i in range(len(L))]
    x=[0 for i in range(len(L))]
    y[0]=l[0][len(l)]

    #Solve for y from L.y=b using forward substitution
    for i in range(1,len(L)):
        sum=0
        for j in range(i):
            sum=sum+L[i][j]*y[j]
        y[i]=(l[i][len(l)]-sum)/L[i][i]
    x[len(L)-1]=y[len(L)-1]/U[len(L)-1][len(L)-1]
    #Solve for x from U.x=y using backward substitution
    for i in range(len(L)-2,-1,-1):
        sum=0
        for j in range(len(L)):
            sum=sum+U[i][j]*x[j]
        x[i]=(y[i]-sum)/U[i][i]
    return(x)


# In[5]:


def Determinant(l):
    U,L = UandL(l)
    detl=1
    for i in range(len(U)):
        detl= -1*detl*U[i][i]
    return detl


#   3. Cholesky Decomposition

# In[7]:


#Decomposition into L and L^T
def Decomposition(l):
    q=[[0 for col in range(len(l))] for row in range(len(l))]      
    for i in range(len(l)):
        for j in range(len(l)):
            if i==j:
                sum=0
                for k in range(i):
                    sum=sum+math.pow(q[k][i],2)
                q[i][j]=math.sqrt(abs(l[i][j]-sum))
            if i<j:
                sum=0
                for k in range(i):
                    sum=sum+q[k][i]*q[k][j]
                q[i][j]=(l[i][j]-sum)/q[i][i]
    return q


# In[8]:


def CholForBack(l,U):
    y=[0 for i in range(len(U))]
    x=[0 for i in range(len(U))]
    y[0]=l[0][len(U)]/U[0][0]

    #Solve for y from L.y=b using forward substitution
    for i in range(1,len(U)):
        sum=0
        for j in range(i):
            sum=sum+U[j][i]*y[j]
        y[i]=(l[i][len(U)]-sum)/U[i][i]
    x[len(U)-1]=y[len(U)-1]/U[len(U)-1][len(U)-1]
    #Solve for x from U.x=y using backward substitution
    for i in range(len(U)-2,-1,-1):
        sum=0
        for j in range(len(U)):
            sum=sum+U[i][j]*x[j]
        x[i]=(y[i]-sum)/U[i][i]
    return x


# 4. Gauss-jacobi

# In[9]:


def checkDiagDom(l):
    for i in range(len(l)-1):
        sum=0
        for j in range(len(l)-1):
            if i!=j:
                sum+=l[i][j]
        if abs((l[i][i]))>abs(sum):
            continue
        else:
            return False
    return True


# In[10]:


def makeDiagDom(A):
    for i in range(len(A)-1):
        max=A[0][i]
        t=i
        for j in range(len(A)-1):
            if abs(A[i][j])>abs(max):
                max=A[i][j]
                t=j
        A[t],A[i]=A[i],A[t]
    return A


# In[11]:


def positiveDefCheck(l):
    c=0
    for i in range(4):
        for j in range(4):
            if l[i][j]!=l[j][i]:
                c+=1
    if c!=0:
        print("Not symmetric hence not positive definite.")
        return False
    else:
        gj=Reduce(l,len(l))
        print(gj)
        c=0
        for i in range(len(l)):
            if l[i][i]<0:
                c+=1
        if c==0:
            print("Positive definite")
            return True
        
        else:
            print("Not positive definite")
            return False
    print(l)


# In[12]:


def Jacobi(l):
    e=0.000001 #initial guess and tolerance
    x_k=[0 for i in range(len(l))]
    x_k1=[0 for i in range(len(l))]
    for i in range(len(l)):
        x_k1[i] = l[i][len(l)-1]/l[i][i]
    n=1
    while (abs(x_k1[0]-x_k[0]))>e or (abs(x_k1[1]-x_k[1]))>e or (abs(x_k1[2]-x_k[2]))>e or (abs(x_k1[3]-x_k[3]))>e or (abs(x_k1[4]-x_k[4]))>e:
        x_k[0],x_k[1],x_k[2],x_k[3],x_k[4]=x_k1[0],x_k1[1],x_k1[2],x_k1[3],x_k1[4]
        for i in range(len(l)):
            sum=0
            for j in range(len(l)):
                if i != j:
                    p=l[i][j]
                    sum = sum + p*x_k[j]
            x_k1[i]=(l[i][len(l)]-sum)/l[i][i]
        n+=1
    return n,x_k1


# 5. Gauss-Seidel

# In[13]:


def GaussSeidel(l):
    x_k=[0 for col in range(len(l))]
    x_k1=[0 for col in range(len(l))]
    e=0.000001#initial guess and tolerance
    for i in range(len(l)):
        sum=0
        for j in range(i):
            sum=sum+(l[i][j])*x_k[j]
        x_k[i] = (l[i][(len(l))]-sum)/l[i][i]
    n=0
    while (abs(x_k1[0]-x_k[0]))>e or (abs(x_k1[1]-x_k[1]))>e or (abs(x_k1[2]-x_k[2]))>e or (abs(x_k1[3]-x_k[3]))>e or (abs(x_k1[4]-x_k[4]))>e:
        x_k1[0],x_k1[1],x_k1[2],x_k1[3],x_k1[4]=x_k[0],x_k[1],x_k[2],x_k[3],x_k[4]
        for i in range((len(l))):
            sum1,sum2=0,0
            for j in range(i):
                sum1 = sum1 + (l[i][j])*x_k[j]
            for m in range(i+1,(len(l))):
                sum2 = sum2 + (l[i][m])*x_k[m]
            x_k[i]=(l[i][(len(l))]-sum1-sum2)/l[i][i]
        #print(x_k1)
        n+=1
    return n,x_k


# 6. Conjugative Gradient method

# In[28]:


def conjugate_gradient(a,b,x,tolerance,iterations):
    a=np.array(a)
    b=np.array(b)
    x=np.array(x)
    # r is the direction of steepest descent
    r=b-(a.dot(x))
    R=r.copy()
    #here R is the new residual for updating purpose
    error=[]
    itr=[]
    for i in range(iterations):
        aR=a.dot(R)
        if np.dot(R, aR)==0:
            break
        alpha=np.dot(R, r)/np.dot(R, aR)
        #alpha is similar to learning rate
        x = x+ (alpha*R)
        r=b-(a.dot(x))
        error.append(np.sqrt(np.sum((r**2))))
        itr.append(i)
        if np.sqrt(np.sum((r**2))) < tolerance:
            break
        else:
            beta= -np.dot(r,aR)/np.dot(R,aR)
            R = r + (beta* R)
    #returning solution x after the process
    return x,error,itr


# In[15]:


def inverse_calculator(a, method,x,tolerance,iterations):
    '''
    x = initial guess of x
    '''
    n = len(a)
    inv = np.zeros((len(a), len(a)))
    #res_list_comb = []
    for i in range(n):
        b = [0.0 for i in range(n)]
        b[i] = 1.0
        X,err,itr = method(a, b, x,tolerance,iterations)
        for j in range(1, len(b)):
            inv[:, i] = X
        #invT.append(X)
        #invT=np.array(invT, dtype=np.double)
        #res_list_comb.extend(res_list)
    #inv = np.transpose(invT)
    return inv,err,itr


def inverse_calculator2(A, method,x,eps):
    n = len(A)
    inv = []
    #res_list_comb = []
    for i in range(n):
        b = [0.0 for i in range(n)]
        b[i] = 1
        X,err,itr = method(A, b, x,eps)
        inv.append(X)
        #res_list_comb.extend(res_list)
    inv = np.transpose(np.array(inv))
    return inv,err,itr


# In[ ]:




