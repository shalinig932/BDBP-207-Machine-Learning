# # Use the above simulated CSV file and implement the following from scratch in Python
# # Read simulated data csv file
# # Form x and y (disease_score_fluct)
# # Write a function to compute hypothesis
# # Write a function to compute the cost
# # Write a function to compute the derivative
# # Write update parameters logic in the main function




X=[[1,1,5],[1,3,9]]
y=[[3],[9]]
learning_rate=0.1
iterations=50



def theta_column(X):
    theta=[]
    for i in range(len(X[0])):
        theta.append([0])
    return theta
theta=theta_column(X)
print(theta)

def hypothesis_gen(sample,theta):
    hypothesis_func=0
    for i in range(len(theta)):
       hypothesis_func+=theta[i][0]*sample[i]
    return hypothesis_func
hypothesis_sample1=hypothesis_gen(X[0],theta)
hypothesis_sample2=hypothesis_gen(X[1],theta)
print(hypothesis_sample1)
print(hypothesis_sample2)



def cost_function(theta,X,y):
    summation=0
    for i in range(len(X)):
        hypothesis_func_for_all_samp = hypothesis_gen(X[i], theta)
        summation+=(hypothesis_func_for_all_samp-y[i][0])**2
    c_f=0.5*summation
    return c_f
cost_func=cost_function(theta,X,y)
print("cost func",cost_func)


def do_updated_theta(theta):
    updated_theta = []
    for i in range(len(theta)):
        updated_theta.append([theta[i][0]])
    return updated_theta

updated_theta = do_updated_theta(theta)
print(updated_theta)

def gradient_function(updated_theta,theta,X,y):
    for j in range(len(theta)):
        summation1=0
        for i in range(len(X)):
            hypothesis_func_for_all_samp = hypothesis_gen(X[i], theta)
            summation1+=(hypothesis_func_for_all_samp-y[i][0])*X[i][j]
        updated_theta[j][0]-=learning_rate*summation1
    return updated_theta
updated_theta_val=gradient_function(updated_theta,theta,X,y)
print("updated theta values:",updated_theta_val)


for i in range(iterations):
    theta_val=gradient_function(updated_theta,theta,X,y)
    theta=theta_val
    print("updated theta values:",theta,"and cost function for it",cost_function(theta,X,y))








