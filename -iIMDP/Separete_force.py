import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import solve_ivp,odeint
from scipy.signal import find_peaks as scipy_find_peaks
class S_sphere:
    def __init__(self,R,E,m,v,y,ro):
        self.R=R
        self.E=E
        self.m=m
        self.v=v
        self.y=1.5*y
        self.ro=ro
        self.c=1e5
        self.Kh = (4 / 3) * self.E * np.sqrt(self.R)
        self.zy = (0.68 * self.R * np.pi ** 2 * self.y ** 2) / (self.E ** 2)
        self.Ky = 1.5 * self.Kh * np.sqrt(self.zy)
        self.zm = self.v * (self.m / self.Ky) ** 0.5
        self.k=0
        en = (v ** -0.25) * 1.324* ((y ** 5) / (ro * E ** 4)) ** 0.125
        er = -np.log(en) / np.sqrt(np.pi ** 2 + np.log(en) ** 2)
        """Adjustable parameter damping constant"""
        self.D=4*er*np.sqrt(self.Kh*self.m)*np.sqrt(R)*2
        print("er:",D/8*np.sqrt(R))

    """vis"""
    def second_order1(self,t,z):
        z0, z1 = z  # z0 = y, z1 = dy/dt
        dz0_dt = z1
        d2y=-(self.D/self.m)*z1*z0**0.5-(self.Kh/self.m)*z0**1.5
        return [z1,d2y]

    def second_order2(self,t,z):
        z0, z1 = z  # z0 = y, z1 = dy/dt
        dz0_dt = z1
        a1=-(np.pi*self.R*(z0-self.zy)*(self.y-self.k*self.zy**0.5)+(4/3)*self.k*np.pi*self.R*(z0**1.5-((z0+self.zy)/2)**1.5)+self.Kh*self.zy**1.5)/self.m
        a2=-(self.D/self.m)*z1*z0**0.5
        d2y=a1+a2
        #d2y=-(self.Ky/self.c)*z1-(self.Ky/self.m)*z0+((self.Ky*self.zy-self.Kh*self.zy**1.5)/self.m)
        return [z1,d2y]

    def second_order3(self,t,z):
        z0, z1 = z  # z0 = y, z1 = dy/dt
        dz0_dt = z1
        a1=-(self.Kh*(z0**1.5-self.zm**1.5)+np.pi*self.R*(self.zm-self.zy)*(self.y-self.k*self.zy**0.5)+(4/3)*self.k*np.pi*self.R*(self.zm**1.5-((self.zm+self.zy)/2)**1.5)+self.Kh*self.zy**1.5)/self.m
        a2=-(self.D/self.m)*z1*z0**0.5
        d2y=a1+a2
        #d2y=-((3*self.Kh)/(2*self.c))*z1*z0**0.5-(self.Kh/self.m)*z0**1.5+alpha/self.m
        return [z1,d2y]

    def run(self):
        y0=[0,self.v]
        t_eval=np.linspace(0,0.1,1000000)
        solution1=solve_ivp(fun=self.second_order1, y0=y0, t_eval=t_eval,t_span=(0,0.1),method='RK45')
        y0 = solution1.y[0]
        y1 = solution1.y[1]
        t=solution1.t
        F=(self.D/self.m)*y1*y0**0.5+(self.Kh/self.m)*y0**1.5
        print("D1:",self.D/self.m)
        peaks, _ = scipy_find_peaks(y0)#Selection the point before the peak value from the first phase
        index=np.argmin(np.abs(y0[:peaks[0]]-self.zy))

        if self.zy>np.max(y0):
            Final=np.interp(np.linspace(0, 0.04, 10000), t, F)
            index4 = np.where(Final < 0)

            if len(index4[0]) > 0:
                Final[index4] = 0
            m = np.linspace(10, 3000, 500)
            test = np.max(Final) * m * np.sqrt(self.Ky / m) * np.exp(
                np.sqrt(self.Ky * m) * np.arctan(
                    self.c * np.sqrt(4 - self.Ky * m / self.c ** 2) / np.sqrt(self.Ky * m)) / (
                        self.c * np.sqrt(4 - self.Ky * m / self.c ** 2))) / self.Ky
            S = np.sum(np.abs(Final)) *(0.04 / 10000)
            mv = test - S
            mv = np.abs(mv)
            index2 = np.argmin(mv)
            return t,F
        else:
            y0 = [self.zy,y1[index]]
            t_eval1 = np.linspace(t[index],0.1,1000000)
            solution1 = solve_ivp(fun=self.second_order2, y0=y0, t_eval=t_eval1, t_span=(t[index], 0.1), method='RK45')
            y00 = solution1.y[0]
            y11 = solution1.y[1]
            a1 = -(np.pi * self.R * (y00 - self.zy) * (self.y - self.k * self.zy ** 0.5) + (
                        4 / 3) * self.k * np.pi * self.R * (
                               y00 ** 1.5 - ((y00 + self.zy) / 2) ** 1.5) + self.Kh * self.zy ** 1.5) / self.m
            a2 = -(self.D/self.m)*y11*y00**0.5
            F0 = -(a1 + a2)*self.m


            Ff0=np.copy(F0)

            t0 = solution1.t

            zm = np.max(y00)
            self.zm=zm
            index1 = np.argmax(y00)
            st = t0[index1]
            F0=F0[:index1]
            t0[0]=0
            F0[0]=0

            y0 = [zm,y11[index1]]
            t_eval1 = np.linspace(st,0.1,1000000)
            solution1 = solve_ivp(fun=self.second_order3, y0=y0, t_eval=t_eval1, t_span=(st, 0.1), method='RK45')
            y000 = solution1.y[0]
            y111 = solution1.y[1]

            t00 = solution1.t
            #plt.plot(t0, y11)
            #plt.plot(t00,y111)
            #plt.show()

            a1 = -(self.Kh * (y000 ** 1.5 - self.zm ** 1.5) + np.pi * self.R * (self.zm - self.zy) * (
                        self.y - self.k * self.zy ** 0.5) + (4 / 3) * self.k * np.pi * self.R * (
                               self.zm ** 1.5 - ((self.zm + self.zy) / 2) ** 1.5) + self.Kh * self.zy ** 1.5) / self.m
            a2 = -(self.D/self.m)*y111*y000**0.5
            F00 = -(a1 + a2)*self.m




            Fall = np.concatenate([F[:index], F0, F00], axis=0)
            tall = np.concatenate([t[:index], t0[:index1], t00], axis=0)
            y = np.interp(np.linspace(0, 0.04, 10000), tall, Fall)
            Final=y

            index4=np.where(Final < 0)
            if len(index4[0])>0:
                Final[index4]=0
            m = np.linspace(10, 3000, 500)
            test = np.max(F0) * m * np.sqrt(self.Ky / m) * np.exp(
                np.sqrt(self.Ky * m) * np.arctan(
                    self.c * np.sqrt(4 - self.Ky * m / self.c ** 2) / np.sqrt(self.Ky * m)) / (
                        self.c * np.sqrt(4 - self.Ky * m / self.c ** 2))) / self.Ky
            S = np.sum(np.abs(Final)) *(0.04 / 10000)
            print("S:",S)
            mv = test - S
            #plt.plot(m,mv)
            #plt.grid()
            #plt.show()
            mv = np.abs(mv)
            index2 = np.argmin(mv)
            m=m[index2]
            en = np.sqrt(m * self.Ky) / (2 * self.c)
            w = np.sqrt(self.Ky / m)
            wd = w * np.sqrt(1 - en ** 2)
            V = np.max(F0) * np.sqrt(w ** 2 + (wd * en) ** 2) / (self.Ky * np.exp(-en * w * np.arctan(wd / (en * w)) / wd))
            t=np.concatenate([t0[:index1],t00])
            F=np.concatenate([F0,F00])
            print(np.shape(t),np.shape(F))
            y = np.interp(np.linspace(0, 0.6, 600), t, F)
            return np.linspace(0, 0.6, 600),y/self.m

class S_flat:
    def __init__(self,R,E,m,v,y,ro):
        self.R=R
        self.E=E
        self.m=m
        self.v=v
        self.y=1.5*y
        self.ro=ro
        self.k=0
        self.dmax=0
        self.R_=self.R
        self.dep=0
        en = (v ** -0.25) * 1.324 * ((y ** 5) / (ro * E ** 4)) ** 0.125
        er = -np.log(en) / np.sqrt(np.pi ** 2 + np.log(en) ** 2)
        """Adjustable parameter damping constant"""
        self.D = 1* er * np.sqrt(2*self.E*self.R * self.m)/self.m

    def second_order1(self,t,z):
        z0, z1 = z  # z0 = y, z1 = dy/dt
        a1=-((2*self.E*z0*self.R-((self.E*z0)**2)/(np.pi*(self.y+self.k*z0**0.5))))/self.m
        a2=-self.D*self.R*z1
        d2y=a1+a2
        return [z1,d2y]
    def second_order2(self,t,z):
        z0, z1 = z  # z0 = y, z1 = dy/dt
        alpha=-2*self.E*self.R_*self.dmax+2*self.E*self.R*self.dmax-((self.E*self.dmax)**2)/(np.pi*(self.y+self.k*self.dmax**0.5))
        a1=-(2*self.E*self.R_*z0+alpha)/self.m
        a2=-self.D*self.R_*z1
        d2y=a1+a2
        return [z1,d2y]

    def second_order22(self,t,z):
        z0, z1 = z  # z0 = y, z1 = dy/dt
        a1=-(np.pi*self.y*self.R**2+(self.k*np.pi*self.R**2)*z0**0.5)/self.m
        a2=-self.D*self.R*z1
        d2y=a1+a2
        return [z1,d2y]

    def run(self):
        self.dep=((np.pi * R * self.k + np.sqrt(
            (np.pi * self.R * self.k) ** 2 + 4 * np.pi * self.R * self.y * self.E)) / (2 * self.E)) ** 2

        y0 = [0, self.v]
        t_eval = np.linspace(0, 0.1, 1000000)
        solution1 = solve_ivp(fun=self.second_order1, y0=y0, t_eval=t_eval, t_span=(0, 0.1), method='RK45')
        y00 = solution1.y[0]
        y11 = solution1.y[1]
        t0 = solution1.t
        a1 = -((2 * self.E * y00 * self.R - ((self.E * y00) ** 2) / (np.pi * (self.y + self.k * y00 ** 0.5)))) / self.m
        a2 = -self.D*self.R*y11
        F11 = -(a1+a2)
        F11[0]=0
        print("dep:",self.dep)


        self.dmax=np.max(y00)

        if self.dmax<self.dep:
            #self.R_ = np.sqrt(self.R ** 2 - ((self.E * self.dmax) / (np.pi * (self.y+self.k*self.dmax**0.5))) ** 2)
            print("Flat 1",self.R_)
            index=np.argmax(y00)
            tm=t0[index]
            y0 = [self.dmax, 0]
            t_eval = np.linspace(tm, 0.1, 1000000)
            solution1 = solve_ivp(fun=self.second_order2, y0=y0, t_eval=t_eval, t_span=(tm, 0.1), method='RK45')
            y0 = solution1.y[0]
            y1 = solution1.y[1]
            t1 = solution1.t
            alpha = -2*self.E*self.R_*self.dmax+2*self.E*self.R*self.dmax-((self.E*self.dmax)**2)/(np.pi*(self.y+self.k*self.dmax**0.5))
            a1 = -(2 * self.E * self.R_ * y0 + alpha) / self.m
            a2 = -self.D*self.R_*y1
            F2 = -(a1 + a2)

            t=np.concatenate([t0[:index],t1])
            F=np.concatenate([F11[:index],F2])
            return t,F

        else:
            peaks, _ = scipy_find_peaks(y00)  # Selection the point before the peak value from the first phase
            if len(peaks)==0:
                index = np.argmin(np.abs(y00 - self.dep))
            else:
                index = np.argmin(np.abs(y00[:peaks[0]] - self.dep))

            tep = t0[index]

            y0 = [self.dep, y11[index]]
            t_eval = np.linspace(tep, 0.1, 1000000)
            solution1 = solve_ivp(fun=self.second_order22, y0=y0, t_eval=t_eval, t_span=(tep, 0.1), method='RK45')
            y0 = solution1.y[0]
            y1 = solution1.y[1]
            t1 = solution1.t
            a1 = -self.D*self.R*y1
            a2 = -(np.pi*self.y*self.R**2+(self.k*np.pi*self.R**2)*y0**0.5) / self.m
            F2 = -(a1 + a2)

            F2[-1]=0
            t = np.concatenate([t0[:index], t1])
            F = np.concatenate([F11[:index], F2])
            return t,F



class S_Cone:
    def __init__(self,theta,E,m,v,y,ro):
        self.E=E
        self.m=m
        self.v=v
        self.y=1.5*y
        self.ro=ro
        self.c=2e5
        self.k=0
        self.dmax=0
        self.theta=theta
        self.ap=0
        en = (v ** -0.25) * (((np.pi * 3 ** 1.25) / (10 * 4 ** 1.25)) ** 0.5) * ((y ** 5) / (ro * E ** 4)) ** 0.125
        er = -np.log(en) / np.sqrt(np.pi ** 2 + np.log(en) ** 2)

        self.D = (2*er*np.sqrt(self.m*((2*self.E)/(np.pi*np.tan(theta)))))/(np.pi*np.tan(theta))

    def second_order1(self,t,z):
        kyc = (4 * self.y) / (np.pi * np.tan(self.theta) ** 2)
        z0, z1 = z  # z0 = y, z1 = dy/dt
        d2y = -((2 * kyc / self.c) * z0 * z1 + (kyc / self.m) * z0 ** 2 + (
                    (0.958 * self.k) / (self.m * np.tan(self.theta) ** 2)) * z0 ** 2.5 + (
                            (2.395 * self.k) / (self.c * np.tan(self.theta) ** 2)) * (z0 ** 1.5) * z1)
        return [z1, d2y]
    def second_order2(self,t,z):
        z0, z1 = z  # z0 = y, z1 = dy/dt
        kyc = (4 * self.y) / (np.pi * np.tan(self.theta) ** 2)
        d2y =-((((4/np.pi)*self.E)/(self.c*np.tan(self.theta)))*z0*z1+(((2/np.pi)*self.E)/(self.m*np.tan(self.theta)))*(z0**2-self.dmax**2)+(kyc / self.m) * self.dmax ** 2+(
                    (0.958 * self.k) / (self.m * np.tan(self.theta) ** 2)) * self.dmax ** 2.5)
        x=(2*self.y)/(self.E*np.tan(self.theta))
        alpha=(2/(np.pi*np.tan(self.theta)))*(self.y/(np.tan(self.theta)*np.cosh(x)**2)+self.E*np.tanh(x))

        d2y=-(((2*alpha)/self.c)*z0*z1+(alpha/self.m)*(z0**2-self.dmax**2)+(kyc / self.m) * self.dmax ** 2+(
                    (0.958 * self.k) / (self.m * np.tan(self.theta) ** 2)) * self.dmax ** 2.5)
        return [z1,d2y]

    def run(self):
        y0 = [0, self.v]
        t_eval = np.linspace(0, 0.2, 1000000)
        solution1 = solve_ivp(fun=self.second_order1, y0=y0, t_eval=t_eval, t_span=(0, 0.2), method='RK45')
        y0 = solution1.y[0]
        y1 = solution1.y[1]
        t0 = solution1.t
        kyc = (4 * self.y) / (np.pi * np.tan(self.theta) ** 2)
        F1=(2*kyc/self.c)*y0*y1+(kyc/self.m)*y0**2+((0.96*self.k)/(self.m*np.tan(self.theta)**2))*y0**2.5+((2.39*self.k)/(self.c*np.tan(self.theta)**2))*(y0**1.5)*y1
        print("coe:",(2/self.c))
        self.dmax=np.max(y0)
        index=np.argmax(y0)
        tmax=t0[index]
        self.ap=F1[index]

        i0 = [self.dmax, 0]
        t_eval = np.linspace(tmax, 0.2, 1000000)
        solution1 = solve_ivp(fun=self.second_order2, y0=i0, t_eval=t_eval, t_span=(tmax, 0.2), method='RK45')
        y00 = solution1.y[0]
        y11 = solution1.y[1]
        t1 = solution1.t
        kyc = (4 * self.y) / (np.pi * np.tan(self.theta) ** 2)
        x = (2 * self.y) / (self.E * np.tan(self.theta))
        alpha = (2 / (np.pi * np.tan(self.theta))) * (
                    self.y / (np.tan(self.theta) * np.cosh(x) ** 2) + self.E * np.tanh(x))
        print("alpha:", alpha / self.E)
        F2 = (((2*alpha)/self.c)*y00*y11+(alpha/self.m)*(y00**2-self.dmax**2)+(kyc / self.m) * self.dmax ** 2+(
                    (0.958 * self.k) / (self.m * np.tan(self.theta) ** 2)) * self.dmax ** 2.5)

        t=np.concatenate([t0[:index],t1])
        F=np.concatenate([F1[:index],F2])

        y = np.interp(np.linspace(0, 0.6, 600), t, F)
        return np.linspace(0, 0.6, 600),y

class S_Cone1:
    def __init__(self,theta,E,m,v,y,ro):
        self.E=E
        self.m=m
        self.v=v
        self.y=1.5*y
        self.ro=ro
        self.k=1e8
        self.dmax=0
        self.theta=theta
        self.ap=0
        en = (v ** -0.25) * 1.324 * ((y ** 5) / (ro * E ** 4)) ** 0.125
        er = -np.log(en) / np.sqrt(np.pi ** 2 + np.log(en) ** 2)#The estimation parameter formula from (Chen et al,2023 Zhang et al,2015)
        """Adjustable parameter damping constant"""
        self.D = 5*(2*er*np.sqrt(self.m*((2*self.E)/(np.pi*np.tan(self.theta)))))/(np.pi*np.tan(self.theta))
        self.x=(2*self.y)/(self.E*np.tan(self.theta))
        self.alpha=(2*self.E*np.tanh(self.x)**2)/(np.pi*np.tan(self.theta))
        self.Kep=(2*self.E*np.tanh(self.x))/(np.pi*np.tan(self.theta))
        self.kyc = (4 * self.y) / (np.pi * np.tan(self.theta) ** 2)
        self.Ke=(2*self.E)/(np.pi*np.tan(self.theta))
        self.KK=(32/(15*np.pi*np.tan(self.theta)**2))*self.k*(1-((1-1/np.cosh(self.x))**1.5)*(1+3/(2*np.cosh(self.x))))
        print("self.kk:",self.KK)

    def second_order1(self,t,z):

        z0, z1 = z  # z0 = y, z1 = dy/dt
        a1=self.Kep*z0**2+(z0**2.5)*(32/(15*np.pi*np.tan(self.theta)**2))*self.k*(1-((1-1/np.cosh(self.x))**1.5)*(1+3/(2*np.cosh(self.x))))
        #a1 = self.kyc * z0 ** 2
        #a1 = self.Ke * z0 ** 2
        a2=self.D*z0*z1
        d2y =-(a1+a2)/self.m
        return [z1, d2y]

    def second_order2(self,t,z):
        z0, z1 = z  # z0 = y, z1 = dy/dt
        A=self.Kep*self.dmax**2+(self.dmax**2.5)*(32/(15*np.pi*np.tan(self.theta)**2))*self.k*(1-((1-1/np.cosh(self.x))**1.5)*(1+3/(2*np.cosh(self.x))))
        #A = self.kyc * self.dmax ** 2
        #A = self.Ke * self.dmax ** 2
        a1=self.Ke*(z0**2-self.dmax**2)+A
        a2=self.D*z0*z1
        d2y=-(a1+a2)/self.m
        return [z1,d2y]

    def run(self):
        y0 = [0, self.v]
        t_eval = np.linspace(0, 0.2, 1000000)
        solution1 = solve_ivp(fun=self.second_order1, y0=y0, t_eval=t_eval, t_span=(0, 0.2), method='RK45')
        y0 = solution1.y[0]
        y1 = solution1.y[1]
        t0 = solution1.t
        a1 = self.Kep * y0 ** 2+(y0**2.5)*(32/(15*np.pi*np.tan(self.theta)**2))*self.k*(1-((1-1/np.cosh(self.x))**1.5)*(1+3/(2*np.cosh(self.x))))
        #a1 = self.kyc * y0 ** 2
        #a1 = self.Ke * y0 ** 2
        a2 = self.D * y0 * y1
        F1 = (a1 + a2) / self.m

        self.dmax=np.max(y0)
        index=np.argmax(y0)
        tmax=t0[index]
        self.ap=F1[index]

        i0 = [self.dmax, 0]
        t_eval = np.linspace(tmax, 0.2, 1000000)
        solution1 = solve_ivp(fun=self.second_order2, y0=i0, t_eval=t_eval, t_span=(tmax, 0.2), method='RK45')
        y00 = solution1.y[0]
        y11 = solution1.y[1]
        t1 = solution1.t
        A = self.Kep*self.dmax**2+(self.dmax**2.5)*(32/(15*np.pi*np.tan(self.theta)**2))*self.k*(1-((1-1/np.cosh(self.x))**1.5)*(1+3/(2*np.cosh(self.x))))
        #A = self.kyc * self.dmax ** 2
        #A = self.Ke * self.dmax ** 2
        a1 = self.Ke * (y00 ** 2 - self.dmax ** 2) + A
        a2 = self.D * y00 * y11
        F2 = (a1 + a2) / self.m
        #plt.plot(t0, F1)
        #plt.plot(t1, F2)
        #plt.show()
        t=np.concatenate([t0[:index],t1])
        F=np.concatenate([F1[:index],F2])
        y = np.interp(np.linspace(0, 0.6, 600), t, F)

        return np.linspace(0, 0.6, 600),y

if __name__=="__main__":

    e1 = 0.3
    e2 = 0.2
    E1 = 0.015e9  # pa
    E2 = 24.366e9  # pa
    E = 1 / ((1 - e1 ** 2) / E1 + (1 - e2 ** 2) / E2)
    y = 1200e3  # pa
    v = 12.95  # m/s
    ro = 1000
    m = 10160
    R=0.4

    Fep=(((4/3)*1e6*R)/3)*((0.1**1.5)/2)
    print("Fep:",Fep)

    en=(v**-0.25)*((np.pi*3**1.25)/(10*4**1.25))*((y**5)/(ro*E**4))**0.125
    er=-np.log(en)/np.sqrt(np.pi**2+np.log(en)**2)
    Kh=E*(4/3)*R**0.5
    D=2*er*np.sqrt(Kh*m)*R**0.5
    print("er:",D)
    print((1.4*200*E**0.4*v**0.2)/(m**(1/3)*ro**(1/15)))

    x = (2 * y) / (E * np.tan(0.6))


    """conical nose"""
    cs=np.linspace(0,1e6,20)
    cmap = cm.get_cmap('viridis')  
    # 从 colormap 中提取 N 种颜色
    N = np.shape(cs)[0]  # 5 条曲线
    colors = [cmap(i / (N - 1)) for i in range(N)]


    for c in range(len(cs)):
        vi=S_Cone1(np.pi/4, E, 10160, v, y, ro)
        vi.k=0
        t,F=vi.run()
        plt.plot(t, F, color=colors[c], alpha=0.8, linewidth=1)
    data = np.loadtxt(r".\Reference data\Pichler2004.txt")
    ti = data[:, 0]
    index = np.argsort(ti)
    ti = np.sort(ti)
    a = -data[index, 1]
    par = 1 / ((np.max(ti) - np.min(ti)) / np.shape(ti)[0])
    a[np.where(a<0)]=0

    print("par:",par)
    plt.plot(ti, a, color="blue", marker="o", alpha=0.8, label=123456789, markersize=3)
    plt.legend()
    plt.show()
    print("Cone error:",(np.max(a)-np.max(F))/(np.max(a)))

    """Flat nose"""
    R = 0.49  # 1
    e1 = 0.3
    e2 = 0.2
    E1 = 0.015e9  # pa
    E2 = 24.366e9  # pa
    E = 1 / ((1 - e1 ** 2) / E1 + (1 - e2 ** 2) / E2)
    y = 950e3  # pa
    v = 17 # m/s
    ro = 2400
    m = 800


    cs = np.linspace(0, 10, 20)*1e6
    cmap = cm.get_cmap('viridis')  # 也可以使用 'plasma', 'rainbow', 'coolwarm' 等
    N = np.shape(cs)[0]  # 5 条曲线
    colors = [cmap(i / (N - 1)) for i in range(N)]
    for c in range(len(cs)):
        vi = S_flat(R, E, m, v, y, ro)
        vi.k=0
        t,F=vi.run()
        plt.plot(t, F, color=colors[c], alpha=0.8, linewidth=1)
    data = np.loadtxt(r".\Reference_data\Gerber2014.txt")
    ti = data[:, 0]
    index = np.argsort(ti)
    ti = np.sort(ti)
    a = data[index, 1]
    par = 1 / ((np.max(ti) - np.min(ti)) / np.shape(ti)[0])
    S = np.sum(np.abs(a*m)) / par

    plt.plot(ti, a, color="cyan", alpha=0.5, linewidth=1, marker="o", label=123456789, markersize=3)
    plt.show()
    print("Flat error:", (np.max(a) - np.max(F)) / (np.max(a)))

    """spherical nose"""
    R = 0.45  # 1
    e1 = 0.3
    e2 = 0.2
    E1 = 0.015e9  # pa
    E2 = 24.366e9  # pa
    E = 1 / ((1 - e1 ** 2) / E1 + (1 - e2 ** 2) / E2)
    y = 1600e3  # pa
    v = 26.7  # m/s
    ro = 2400
    m = 850


    cs = np.linspace(0, 10, 20) * 1e6
    cmap = cm.get_cmap('viridis')  # 也可以使用 'plasma', 'rainbow', 'coolwarm' 等
    N = np.shape(cs)[0]  # 5 条曲线
    colors = [cmap(i / (N - 1)) for i in range(N)]
    for c in range(len(cs)):
        vi = S_sphere(R, E, m, v, y, ro)
        vi.k = cs[c]
        t, F = vi.run()
        plt.plot(t, F, color=colors[c], alpha=0.8, linewidth=1)
    data = np.loadtxt(r".\Reference data\CALVETTI2012.txt")
    t = data[:, 0]
    index = np.argsort(t)
    ti = np.sort(t)
    a = data[index, 1] * 1000
    par = 1 / ((np.max(ti) - np.min(ti)) / np.shape(ti)[0])
    S = np.sum(np.abs(a)) / par
    plt.show()

    S1=np.linspace(0,1e4,40)
    for i in S1:
        plt.scatter([i / S], [(V - v) / v], color="red", alpha=0.5)
        plt.scatter([i/S],[(m-850)/850],color="orange",alpha=0.5)
    plt.scatter([],[],color="red",label="123456")
    plt.scatter([], [], color="orange", label="123456")
    plt.legend()
    plt.grid(linestyle="--")
    plt.show()

    plt.plot(ti, a/m , color="grey", alpha=0.5, linewidth=1, marker="o", label=123456789, markersize=3)

    plt.show()
    print("Sphere error:", (np.max(a/m) - np.max(F)) / (np.max(a/m)))