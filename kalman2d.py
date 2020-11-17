import numpy as np
import matplotlib.pyplot as plt
observed_x=[]
kalman_x=[]
observed_y=[]
kalman_y = []
def kalman_xy(x, P, measurement, R,
              motion = np.matrix('0. 0. 0. 0.').T,
              Q = np.matrix(np.eye(4))):
    """
    Parameters:    
    x: initial state 4-tuple of location and velocity: (x0, x1, x0_dot, x1_dot)
    P: initial uncertainty convariance matrix
    measurement: observed position
    R: measurement noise 
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    """
    return kalman(x, P, measurement, R, motion, Q,
                  F = np.matrix('''
                      1. 0. 1. 0.;
                      0. 1. 0. 1.;
                      0. 0. 1. 0.;
                      0. 0. 0. 1.
                      '''),
                  H = np.matrix('''
                      1. 0. 0. 0.;
                      0. 1. 0. 0.'''))

def kalman(x, P, measurement, R, motion, Q, F, H):
    '''
    Parameters:
    x: initial state
    P: initial uncertainty convariance matrix
    measurement: observed position (same shape as H*x)
    R: measurement noise (same shape as H)
    motion: external motion added to state vector x
    Q: motion noise (same shape as P)
    F: next state function: x_prime = F*x
    H: measurement function: position = H*x

    Return: the updated and predicted new values for (x, P)


    This version of kalman can be applied to many different situations by
    appropriately defining F and H 
    '''
    # UPDATE x, P based on measurement m    
    # distance between measured and current position-belief
    y = np.matrix(measurement).T - H * x
    S = H * P * H.T + R  # residual convariance
    K = P * H.T * S.I    # Kalman gain
    x = x + K*y
    I = np.matrix(np.eye(F.shape[0])) # identity matrix
    P = (I - K*H)*P

    # PREDICT x, P based on motion
    x = F*x + motion
    P = F*P*F.T + Q

    return x, P

def demo_kalman_xy():
    x = np.matrix('0. 0. 0. 0.').T 
    P = np.matrix(np.eye(4))*1000 # initial uncertainty

    N = 10
 

#     true_x = [1.375,
# 5.925, 
# 9.575, 
# 12.875,  
# 16.375,  
# 20.375,  
# 27.125,  
# 31.275, 
# 35.825, 
# 41.875]

    # true_y = [0.7, 3.05, 5.1, 8.65,11,14.5,16.2,17.15, 18.3, 18.8]

    true_x = [ 0.675 ,
        0.575 ,
        0.625 ,
        0.625 ,
        0.425 ,
        0.425 ,
        0.425 ,
        0.525 ,
        0.575 ,
        0.575 ,
        10.475 ,
        20.425 ,
        30.625 ,
        40.425 ,
        50.325 ,
        60.475 ,
        70.475 ,
        80.275 ,
        90.575 ,
        100.425 ,
        100.425 ,
        100.425 ,
        100.475 ,
        100.575 ,
        100.425 ,
        100.625 ,
        100.475 ,
        100.475 ,
        100.475 ,
        90.675 ,
        80.725 ,
        70.675 ,
        60.625 ,
        50.475 ,
        40.775 ,
        30.775 ,
        30.025 ,
        10.825 ,
        10.825 ,
        10.475 ,
        10.475 ,
        10.525 ,
        10.525 ,
        10.525 ,
        10.425 ,
        10.525 ,
        20.575 ,
        30.625 ,
        40.625 ,
        50.325 ,
        60.125 ,
        70.425 ,
        80.525 ,
        90.275 ,
        90.525 ,
        90.525 ,
        90.475 ,
        90.525 ,
        90.525 ,
        80.475 ,
        70.675 ,
        60.425 ,
        50.825 ,
        40.725 ,
        30.625 ,
        20.775 ,
        20.475 ,
        20.475 ,
        30.375 ,
        40.475 ,
        50.675 ,
        60.525 ,
        70.425 ,
        70.375 ,
        70.375 ,
        70.275 ,
        60.475 ,
        50.575 ,
        40.775 ,
        40.275 ,
        50.475 ]

   
    true_y = [  0.5 ,
         10.5 ,
         20.35 ,
         30.35 ,
         40.6 ,
         50.55 ,
         60.2 ,
         70.7 ,
         80.6 ,
         90.55 ,
         90.5 ,
         90.3 ,
         90.3 ,
         90.3 ,
         90.4 ,
         90.55 ,
         90.6 ,
         90.55 ,
         90.5 ,
         90.5 ,
         80.65 ,
         70.75 ,
         60.45 ,
         50.5 ,
         40.5 ,
         30.35 ,
         20.5 ,
         10.75 ,
         00.7 ,
         0.55 ,
         0.4 ,
         0.45 ,
         0.5 ,
         0.5 ,
         0.5 ,
         0.5 ,
         0.4 ,
         0.4 ,
         10.45 ,
         20.35 ,
         30.25 ,
         40.45 ,
         50.4 ,
         60.5 ,
         70.45 ,
         80.3 ,
         80.5 ,
         80.5 ,
         80.5 ,
         80.45 ,
         80.45 ,
         80.6 ,
         70.65 ,
         60.4 ,
         50.8 ,
         40.6 ,
         30.75 ,
         20.6 ,
         10.5 ,
         20.45 ,
         10.55 ,
         10.5 ,
         10.5 ,
         10.5 ,
         20.25 ,
         30.35 ,
         40.65 ,
         50.3 ,
         50.4 ,
         60.4 ,
         60.45 ,
         60.5 ,
         60.5 ,
         50.65 ,
         40.65 ,
         30.7 ,
         20.5 ,
         20.4 ,
         30.45 ,
         40.55 ,
         40.45 ]
    
    # true_x= np.random.random_integers(1,100,N)
    
    # true_x = np.linspace(0.0, 100.0, N)
    # true_y = true_x
    # true_y = np.random.random_integers(1,100,N)
    true_x = [10,20,30,40,50,60,70,80,90,100]
    true_y = [10,15,13,16,25,35,25,28,32,23]
    true_x = [4.325, 
    60.175,     90.025, 
    80.375,     10.325, 
    12.275,     11.225, 
    14.225,     16.825, 
    17.175,     19.075, 
    20.325,     21.775, 
    21.425,     23.475, 
    25.375,     27.525, 
    30.175,     27.775, 
    24.775,    22.725, 
    26.075,     30.125, 
    32.775,     33.675, 
    31.975,     33.425, 
    31.125,     30.175, 
    35.525,     36.125, 
    38.125,    39.175, 
    38.375,    41.425,
    42.375,    44.225, 
    43.475,    43.675,
    46.325,    45.975,    48.775, 
    47.825,    46.675, 
    45.075,    42.775, 
    40.725,     51.175,     52.075,     54.175,     54.375,     53.475,     50.675, 
    49.825,    54.425, 
    56.275,    53.975, 
    50.675,     47.775, 
    57.925,     58.375,     59.675,     59.775, 
    57.825,     56.875, 
    61.475,     64.025, 
    62.525,     60.325, 
    64.975,     65.375, 
    66.625,     67.825,    68.725,    67.175,    68.375,    63.125,    65.825] 

    true_y = [
    40.35,     60.35, 
    60.75,    90.75, 
    10.55,    12.35, 
    14,     15.1, 
    14.8,    17, 
    20.55,    19.5, 
    20.95,    23.2, 
    23,     26.05, 
    31.75,    29.5,     28.45, 
    27.1,     26.05, 
    30,     32.1,
    39.6,    37.35,
    35.4,    34.1,
    31.85,    35.05,
    37.3,     41, 
    42.95,     44.3, 
    46.35,     47.15, 
    49.9,     52.05, 
    52.65,    48.85, 
    56.5,     54.8, 
    59.35,    56.45,
    52,     50.7, 
    47.9,    45.35, 
    61.95,    64.15, 
    66.8,     65.35, 
    59.55,    57.1, 
    55.65,    64.65, 
    65,     62.15, 
    57.3,    53.2, 
    68.55,    67.35, 
    67.95,    69.4, 
    63.2,    68.2, 
    68.6,    73.6, 
    70.25,    70.75, 
    73.15,     76.4, 
    78.55,     79.75, 
    82.15,     82, 
    80.5,     73.05, 
    74.9    ]

    true_x = [80.625,
    80.525,
    80.425,
    70.675,
    70.375,
    70.475,
    70.475,
    70.525,
    70.525,
    70.625,
    60.325,
    50.375,
    40.425,
    30.525,
    20.525,
    20.425,
    20.425,
    10.325,
    10.525,
    10.525,
    10.425,
    10.575,
    10.625,
    20.675,
    30.325,
    40.225,
    40.375,
    40.375,
    40.375,
    40.375,
    40.375,
    50.325,
    50.325,
    50.325,
    50.425,
    50.325,
    50.475,
    50.475,
    50.675,
    40.425,
    30.925,
    20.525,
    10.725,
    0.725,
    0.625,
    0.625,
    0.625,
    0.625,
    0.625,
    0.625,
    0.525,
    0.475,
    0.475
        ]

    true_y =[0.45,
    10.5 ,
    20.5 ,
    30.5 ,
    40.5 ,
    50.65,
    60.35,
    70.4 ,
    80.45,
    90.5 ,
    90.6 ,
    90.65,
    90.5 ,
    90.5 ,
    90.4 ,
    80.75,
    70.35,
    70.45,
    60.5 ,
    50.5 ,
    40.5 ,
    30.45,
    20.45,
    20.6 ,
    20.6 ,
    20.55,
    30.55,
    40.35,
    50.4 ,
    60.35,
    70.4 ,
    70.45,
    60.55,
    50.8 ,
    40.35,
    30.55,
    20.55,
    10.65,
    0.6 ,
    0.55,
    0.55,
    0.55,
    0.65,
    0.65,
    10.55,
    20.75,
    30.6 ,
    40.6 ,
    50.3 ,
    60.4 ,
    70.4 ,
    80.5 ,
    90.4 ]
    # observed_x = true_x + 0.05*np.random.random(N)*true_x
    # observed_y = true_y + 0.05*np.random.random(N)*true_y
    # V=0.1
    n = 50
    n2 = n*10.4
    V=n/n2
    true_x  = np.arange(0.0,n,V)
    #* np.pi / 180.
    true_y = np.sin(true_x)
    # true_x = np.linspace(0,V,n)
    # true_y =true_x*2
    observed_x = true_x
    observed_y = true_y
    # print("ox",observed_x,'\n',"oy",observed_y)
    # plt.plot(observed_x, observed_y, 'ro',markersize=12,label="Observed",alpha=1)
    plt.plot(observed_x,observed_y,'r-',markersize=12,alpha=0.5)

    result = []
    R = 0.02**2
    # R = 100

    for meas in zip(observed_x, observed_y):
        x, P = kalman_xy(x, P, meas, R)
        result.append((x[:2]).tolist())
    kalman_x, kalman_y = zip(*result)
    # print("kx",kalman_x,'\n',"ky",kalman_y)
    # plt.plot(kalman_x, kalman_y, 'go',markersize=12,label="Kalman Prediction",alpha=1)
    plt.plot(kalman_x,kalman_y,'b-',markersize=12,alpha=0.5)

    # mean_x = np.array(abs((kalman_x + observed_x)/2))
    # mean_y = np.array(abs((kalman_y + observed_y)/2))
    # # print("x",mean_x.shape)
    # # print("y",mean_y.shape)
    kalman_x =list(kalman_x)
    kalman_y =list(kalman_y)
   
    kalman_x = [val for sublist in kalman_x for val in sublist]
    kalman_y = [val for sublist in kalman_y for val in sublist]

    observed_x = np.array(observed_x)
    kalman_x =np.array(kalman_x)
    observed_y = np.array(observed_y)
    kalman_y =np.array(kalman_y)

    # print("ox",observed_x,'\n',"oy",observed_y)
    # print("kx",kalman_x,'\n',"ky",kalman_y)
    # mx = np.reshape(mean_x,(1,400))
    # my = np.reshape(mean_y,(1,400))
    # ax=(observed_x-kalman_x)
    # ay =(observed_y-kalman_y)
    # plt.plot(ax,ay,'bo')
    # print("ax",ax,"ay",ay)
    # print("x",mx,'\n',"y",my)
    # plt.plot(mx,my,'r*')

 
    accpred =0
    inacc =0
    # xper = (observed_x / kalman_x * 100)
    # yper =(observed_y / kalman_y * 100)
    
    # m = np.sqrt((kalman_x-observed_x)**2 + (kalman_y-observed_y)**2)
    diff = np.sqrt((kalman_x-observed_x)**2 + (kalman_y-observed_y)**2)
    # diff = np.array(abs((observed_x-kalman_x)))
    diff = [float(i) for i in diff]
    valued = np.count_nonzero(true_x)
    print("caount",valued)

    # if valued >= 10:
        #default 11 10.02
    threshold = V+V/2.471
        # 0.041
    # if N >= 10:
    #     threshold =15
    # else:
    #     threshold =valued *0.5
    # threshold =11
    print(diff)
    for i in diff:
        print(i)
        if i > threshold:
            inacc = inacc+1

        else:
            accpred = accpred+1
    print("Accurate prediciton:",accpred)
    print("Inaccurate prediction",inacc)
    diffper = inacc/(inacc+accpred)*100
    # m=(xper+yper)/2
    # m = [float(i) for i in m]
    # print("precent",m,"%")
    diffperstr =str(diffper)
    # mena = np.mean(m)
    # mei =str(round(mena,3))+"%"

    # print("Mean Percentage of error:",mena)
    print("Mean Percentage of error:",diffperstr)

    plt.xlabel("x- axis", fontdict=None, labelpad=None)
    plt.ylabel("y- axis", fontdict=None, labelpad=None)
    # plt.text(20,10,"Mean Percentage error:"+mei)
    R=str(round(R,6))
    plt.text(20,10,"Noise Ratio:"+R)

    # plt.title("Mean Percentage error:"+mei)
    per = 100 - diffper
    per=str(round(per,6))
    per = str(per)
    plt.title("Prediction Accuracy:"+per+"%")
    plt.grid(b=None,which='major',axis='both',)
    

    plt.ylim(-1,1.5)
    plt.xlim(0,5)    

    
    plt.legend()
    plt.show()


demo_kalman_xy()

