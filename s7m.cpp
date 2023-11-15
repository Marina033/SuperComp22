#include <iostream>
#include <fstream>
#include <math.h>
#include <time.h>
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
#define Ar(i,j) Ar[(i)*(N+1)+j]

int main(int argc,char *argv[])
{
    double *a = new double [(M+1)*(N+1)]();
    double *b = new double [(M+1)*(N+1)]();
    double *F = new double [(M+1)*(N+1)]();
    double *w = new double [(M+1)*(N+1)]();
    double *r = new double [(M+1)*(N+1)]();
    double *Ar = new double [(M+1)*(N+1)]();

    double x1=-1, x2=1, y1=-0.5, y2=0.5;   // границы фиктивной области
    double h1=(x2-x1)/M, h2=(y2-y1)/N;     // шаги сетки по x и по y
    double eps=max(h1,h2)*max(h1,h2);
    
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
            double l=(yb-ya)/h2;  // доля отрезка y[j-1/2] - y[j+1/2] попавшая внутрь эллипса
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
            double l=(xb-xa)/h1;  // доля отрезка x[i-1/2] - x[i+1/2] попавшая внутрь эллипса
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
            int n=100;                                   // 
            double dx=(xb-xa)/n;                         // площадь ячейки
            double S=0;                                  // x[i-1/2]-x[i+1/2], y[j-1/2] - y[j+1/2]
            for (int k=0; k<n; k++)                      // определяется интегралом методом
            {                                            // прямоугольников
                double x=xa+(k+0.5)*dx;
                double el=0.5*sqrt(1-x*x);
                S += dx*max(0.0,min(yb,el)-max(ya,-el));
            }
            F(i,j)=S/h1/h2;
        }
    }

    // итерации метода взвешенных невязок
    int it=0;
    double dw;
    for (it=0;it<10000000; it++)
    {
        // вычисление вектора невязки r = A w - F
        //#pragma omp parallel for shared(r)
        #pragma omp parallel for shared(a,b,F,w,r,h1,h2)
        for (int i=1; i<M; i++)
        {
            for (int j=1; j<N; j++)
            {
                r(i,j)=-F(i,j);
                r(i,j) -= (a(i+1,j)*(w(i+1,j)-w(i,j)) - a(i,j)*(w(i,j)-w(i-1,j)))/h1/h1;
                r(i,j) -= (b(i,j+1)*(w(i,j+1)-w(i,j)) - b(i,j)*(w(i,j)-w(i,j-1)))/h2/h2;
            }
        }

        // вычисление вектора A r
        //#pragma omp parallel for shared(Ar)
        #pragma omp parallel for shared(a,b,F,Ar,r,h1,h2)
        for (int i=1; i<M; i++)
        {
            for (int j=1; j<N; j++)
            {
                Ar(i,j)=0.0;
                Ar(i,j) -= (a(i+1,j)*(r(i+1,j)-r(i,j)) - a(i,j)*(r(i,j)-r(i-1,j)))/h1/h1;
                Ar(i,j) -= (b(i,j+1)*(r(i,j+1)-r(i,j)) - b(i,j)*(r(i,j)-r(i,j-1)))/h2/h2;
            }
        }

        // вычисление ( Ar, Ar), ( Ar, r), (r, r)
        double Ar2=0.0;
        double Arr=0.0;
        double r2=0.0;
        #pragma omp parallel for reduction (+:Ar2,Arr,r2)
        for (int i=0; i<(M+1)*(N+1); i++)
        {
             Ar2 += Ar[i]*Ar[i];
             Arr += Ar[i]*r[i];
             r2 += r[i]*r[i];
        }

        double tau=Arr/Ar2;
        for (int i=0; i<nthr; i++) dwthr[i]=0;
        // вычисление w = w - (Ar, r)/(Ar, Ar) r
        #pragma omp parallel for shared(w,r,dwthr)
        for (int i=0; i<(M+1)*(N+1); i++)
        {
            w[i] -= tau*r[i];
            // максимум изменения w для каждого треда
            int id = omp_get_thread_num();
            if ( dwthr[id] < abs(tau*r[i]) ) dwthr[id] = abs(tau*r[i]);
        }

        dw=0;
        for (int i=0; i<nthr; i++) dw=max(dw,dwthr[i]); // максимум изменения w на итерации
        //dw = tau*sqrt(h1*h2*r2);       // норма изменения решения на итерации
        if (sqrt(h1*h2*r2)<0.5) break;     // окончание итераций при достижении точности
    }

        
    cout << "MY TIME OF PROG = " << omp_get_wtime() - time << endl;
    cout << "It = " << it << ", dw = " << dw << endl;

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

    delete a;
    delete b;
    delete F;
    delete w;
    delete r;
    delete Ar;
}
