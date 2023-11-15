#include <iostream>
#include <fstream>
#include <math.h>
#include <omp.h>
#include <cstdlib>
#include <cmath>
using namespace std;  

#define N 80
#define M 80
#define a(i,j) a[(i)*(N+1)+j]
#define b(i,j) b[(i)*(N+1)+j]
#define w(i,j) w[(i)*(N+1)+j]
#define F(i,j) F[(i)*(N+1)+j]
#define r(i,j) r[(i)*(N+1)+j]

int main(int argc,char *argv[])
{
    double *a = new double [(M+1)*(N+1)]();
    double *b = new double [(M+1)*(N+1)]();
    double *F = new double [(M+1)*(N+1)]();
    double *w = new double [(M+1)*(N+1)]();
    double *r = new double [(M+1)*(N+1)]();
    double x1=-1, x2=1, y1=-0.5, y2=0.5;   // границы фиктивной области
    double h1=(x2-x1)/M, h2=(y2-y1)/N;     // шаги сетки по x и по y
    double eps=h1*h2;

    int nthr=atoi(argv[1]);
    omp_set_num_threads(nthr);
    double *dwthr = new double [nthr]();
    
    double time = omp_get_wtime();

    // вычисление коэффициентов a_ij
    #pragma omp parallel for shared(a,x1,h1,y1,h2,eps)  //private(i,j,x,el,ya,yb,l)
    for (int i=1; i<=M; i++)
    {
        double x=x1+(i-0.5)*h1;
        double el=0.5*sqrt(1-x*x);
        for (int j=1; j<=N; j++)
        {
            double ya=y1+(j-0.5)*h2;
            double yb=ya+h2;
            if (yb>=-el) yb=min(el,yb);
            else   // отрезок [ya,yb] ниже реальной области (эллипса)
            {
               a(i,j)=1.0/eps;   
               continue;
            }
            if (ya<=el) ya=max(-el,ya);
            else    // отрезок [ya,yb] выше реальной области (эллипса)
            {
               a(i,j)=1.0/eps;
               continue;
            }
            double l=(yb-ya)/h2;
            a(i,j) = l+(1-l)/eps;
        }
    }

    // вычисление коэффициентов b_ij
    #pragma omp parallel for shared(b,x1,h1,y1,h2,eps)
    for (int j=1; j<=N; j++)
    {
        double y=y1+(j-0.5)*h2;
        double el=sqrt(1-4*y*y);
        for (int i=1; i<=M; i++)
        {
            double xa=x1+(i-0.5)*h1;
            double xb=xa+h1;
            if (xb>=-el) xb=min(el,xb);
            else   // отрезок [xa,xb] левее реальной области (эллипса)
            {
               b(i,j)=1.0/eps;   
               continue;
            }
            if (xa<=el) xa=max(-el,xa);
            else    // отрезок [xa,xb] правее реальной области (эллипса)
            {
               b(i,j)=1.0/eps;
               continue;
            }
            double l=(xb-xa)/h1;
            b(i,j) = l+(1-l)/eps;
        }
    }

    // вычисление правой части F_ij
    #pragma omp parallel for shared(F,x1,h1,y1,h2)
    for (int i=1; i<M; i++)
    {
        double xa=x1+(i-0.5)*h1;
        double xb=xa+h1;
        for (int j=1; j<N; j++)
        {
            double ya=y1+(j-0.5)*h2;
            double yb=ya+h2;
            int n=100;
            double dx=(xb-xa)/n;
            double S=0;
            for (int k=0; k<n; k++)
            {
                double x=xa+(k+0.5)*dx;
                double el=0.5*sqrt(1-x*x);
                S += dx*max(0.0,min(yb,el)-max(ya,-el));
            }
            F(i,j)=S/h1/h2;
        }
    }

    // итерации Якоби
    int it=0;
    double dwmax=0;
    for (it=0;it<1000000; it++)
    {
        // вычисление вектора
        // r_ij = F_ij - A_{i+1,j} w_{i+1,j} - A_{i-1,j} w_{i-1,j} - A_{i,j+1} w_{i,j+1} - A_{i,j-1} w_{i,j-1}
        #pragma omp parallel for shared(a,b,F,w,r,h1,h2)
        for (int i=1; i<M; i++)
        {
            for (int j=1; j<N; j++)
            {
                r(i,j) =  F(i,j);
                r(i,j) += (a(i+1,j)*w(i+1,j) + a(i,j)*w(i-1,j))/h1/h1;
                r(i,j) += (b(i,j+1)*w(i,j+1) + b(i,j)*w(i,j-1))/h2/h2;
            }
        }

        // изменение текущего приближения решения w
        // w_ij = r_ij /A_ij
        for (int i=0; i<nthr; i++) dwthr[i]=0;
        #pragma omp parallel for shared(a,b,r,w,h1,h2,dwthr)
        for (int i=1; i<M; i++)
        {
            for (int j=1; j<N; j++)
            {
                double wo=w(i,j);
                w(i,j) = r(i,j)/((a(i+1,j) + a(i,j))/h1/h1 + (b(i,j+1) + b(i,j))/h2/h2);
                double dw=abs(w(i,j)-wo);
                // максимум изменения w для каждого треда
                int id = omp_get_thread_num();
                if ( dwthr[id] < dw ) dwthr[id] = dw;
            }
        }

        dwmax=0;
        for (int i=0; i<nthr; i++) dwmax=max(dwmax,dwthr[i]); // максимум изменения w по потокам
        if (dwmax<1e-6) break;    // выход из цикла итераций Якоби при достижении точности
    }
    
    cout << "MY TIME OF PROG = " << omp_get_wtime() - time << endl;
    cout << "It = " << it << ", dw = " << dwmax << endl;

    // вывод решения в файл
    ofstream rslt;
    rslt.open ("result.txt");
    for (int j=0; j<=N; j++)
    {
        rslt << w(0,j);
        for (int i=1; i<=M; i++)
            rslt << ",\t" << w(i,j);
        rslt << endl;
    }
	
    rslt.close();
    delete[] a;
    delete[] b;
    delete[] F;
    delete[] w;
    delete[] r;
}
