
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from math import isclose
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

class consumer_min:
    
    def __init__(self,**kwargs): # called when created
        
        # a. baseline parameters
        self.alpha = 0.5
        
        self.p1 = 1
        self.p2 = 2
        self.u = 10
        self.e = np.nan
        
        self.x1 = np.nan # not-a-number
        self.x2 = np.nan
        
        # b. baseline settings
        self.x2_max = 20
        self.N = 100
            
        # c. update parameters and settings
        for key, value in kwargs.items():
            setattr(self,key,value) # like self.key = value
        
         # note: "kwargs" is a dictionary with keyword arguments
            
    def __str__(self): # called when printed
        
        lines = f'Alpha = {self.alpha:.3f}\n'
        lines += f'Price vector = (p1,p2) = ({self.p1:.3f},{self.p2:.3f})\n'
        lines += f'Utility = u = {self.u:.3f}\n'
        
        # add lines on solution if it has been calculated
        if not (np.isnan(self.x1) or np.isnan(self.x2)):
            lines += 'Solution:\n'
            lines += f' x1 = {self.x1:.2f}\n'
            lines += f' x2 = {self.x2:.2f}\n'
        if not np.isnan(self.e):
            lines += f"Expenditure = {self.e:.2f}\n"

        # add error term if x2_max is to small to find optimal solution
        # isclose-function is used to find small discrepancies 
        if isclose(self.x2, self.x2_max ) or isclose(self.x1, self.x2_max) or self.u < self.u_func(self.x1,self.x2):
            lines += f"Error: optimal solution possible not found, axis length is to small"
   
        return lines

        # note: \n gives a lineshift

    # utilty function
    def u_func(self,x1,x2):
        return x1**self.alpha*x2**(1-self.alpha)
    # expense function
    def expense(self,x1,x2):
        return x1*self.p1+x2*self.p2

    
    # solve problem
    def solve(self):
        
        # a. objective function (to minimize) 
        def expense(x):
            return self.expense(x[0],x[1])
        
        # b. constraints
        constraints = ({'type': 'ineq', 'fun': lambda x: self.u_func(x[0],x[1])-self.u})
        bounds = ((0,self.x2_max),(0,self.x2_max))
        
        # c. call solver
        initial_guess = [self.u*self.alpha,self.u*self.alpha]
        sol = optimize.minimize(expense,initial_guess,
                                method='SLSQP',bounds=bounds,constraints=constraints)
        
        # d. save
        self.x1 = sol.x[0]
        self.x2 = sol.x[1]
        self.e = self.expense(self.x1,self.x2)
        self.u = self.u_func(self.x1,self.x2)
  
    
    # find and plot budgetlines
    def plot_budgetlines(self,ax):
        # allocate memory
        self.x1_buds = []
        self.x2_buds = []
        self.es = []

        for fac in [0.75,1,1.25]:
            e = fac * self.expense(self.x1,self.x2)
            x = [0,e/self.p1] # x-cordinates in triangle
            y = [e/self.p2,0] #y-coordiates in traingle
            self.x1_buds.append(x)
            self.x2_buds.append(y)
            self.es.append(e)

        # fill triangle
        for x,y,e in zip(self.x1_buds,self.x2_buds,self.es):
            ax.plot(x,y, label = f"$E = {e:.2f} $")
        
    # plot solution
    def plot_solution(self,ax):
        
        ax.plot(self.x1,self.x2,'ro',color='black') # a black dot
        ax.text(self.x1*1.03,self.x2*1.03,f'$E^{{min}} = {self.e:.2f}$')
        
    # plot indifference curve
    def plot_indifference_curves(self,ax):
        
        u = self.u_func(self.x1,self.x2)
            
            # b. allocate numpy arrays
        x1_vec = np.empty(self.N)
        x2_vec = np.linspace(1e-8,self.x2_max,self.N)
        
        #find plots
        for i,x2 in enumerate(x2_vec):
            def objective(x1):
                return self.u_func(x1,x2)-u
            sol = optimize.root(objective, 0.1)
            x1_vec[i] = sol.x[0]
    
        # d.plot
        ax.plot(x1_vec,x2_vec,label=f'$u = {u:.2f}$')
        ax.fill_between(x1_vec,x2_vec,self.x2_max, color="firebrick",lw=2,alpha=0.5)
    
    # details of the plot (label,limits,grid,legend)
    def plot_details(self,ax):

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
                
        ax.set_xlim([0,self.x2_max])
        ax.set_ylim([0,self.x2_max])

        ax.grid(ls='--',lw=1)
        ax.legend(loc='upper right')

# function that executes all of the class functions
def solve_and_plot(u,alpha, p1 , x2_max):
    # defines consumer
    hans = consumer_min(alpha = alpha, p1 = p1, p2 = 1, x2_max=x2_max, u = u)
    # solves the problem using module:
    hans.solve()
    # creates figure:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    #plots the solution using module
    hans.plot_budgetlines(ax)
    hans.plot_indifference_curves(ax)
    hans.plot_solution(ax)
    hans.plot_details(ax)
    
    #prints detail
    print(hans)

#Plots an interactive version of the solve_and_plot
def make_interactive():
    return interact(solve_and_plot, u=widgets.IntSlider(min=1,max=20,step=1,value=10), \
         alpha=widgets.FloatSlider(description = "$\\alpha$", min=0.05,max=0.96,step=0.05,value=0.3),\
         p1 =widgets.FloatSlider(description = "$\\frac{p_1}{p_2}$" ,min=0.1,max=10,step=0.1, value=2), \
         x2_max=widgets.IntSlider(description = "Axis length", min=5,max=40,step=5,value=25))
# widget.FLoatSlider and .IntSlider defines starting values and intervals for floats and ints, so it dosen't crash for for examlpe u=0