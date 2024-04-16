import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def Data_Import(path):
        try:
            #Data from investing.com
            df = pd.read_csv(path, thousands=',')
            btcDailyHigh = df['High'].tolist()
            time_stamps = df['Date'].tolist()
            #df = pd.read_excel('data/bitcoin/bitcoin_2022-2023.xlsx', engine='openpyxl')
            #btcDailyHigh = df['priceHigh'].tolist()
            #time_stamps = df['timeHigh'].tolist()
            btcDailyHigh.reverse()
            time_stamps.reverse()
            return time_stamps, btcDailyHigh
        except Exception as e:
            print('Error while opening {}: '.format(path)+str(e))
            return [], []


def Normalize(data):
    return [(x-min(data))/(max(data)-min(data)) for x in data]

def Denormalize(data, min, max):
    denorm_data = []
    for x in data:
        if x:
            denorm_data.append(min+x*(max-min))
        else:
            denorm_data.append(0)
    return denorm_data

def Eucl_Dist(A,B):
    if(len(A)!=len(B)):
        return 9999.0
    else:
        try:
            d_sq = 0
            for i in range(len(A)):
                d_sq = d_sq + abs(A[i]-B[i])**2
            return d_sq**0.5
        except Exception as e:
            print('Error while computing the distance: {}'.format(str(e)))
            return -1


def Data_Samples(data,t,L,K_max):
    try:
        Y_train = data[:t+1]
        Y_train_norm = Normalize(Y_train)
        patterns = []
        #L-1 loops
        #Generate all patterns for L and K_max, length=K_max^(L-1)
        for i in range(1, K_max+1):
            for j in range(1, K_max+1):
                for k in range(1, K_max+1):
                    patterns.append([i,j,k])
        #Generate data_samples from Y_train
        #For the sake of simplicity, assuming a pattern alpha = [k1,k2,k3]
        #we use the z-vectors (motifs) [t,t-k3,t-k2-k3,t-k1-k2-k3], etc...
        #No clustering over all the z-vectors for each alpha
        z_vectors=[]
        for alpha in patterns:
            alpha_z_vectors = []
            index = 0
            while(index+sum(alpha)<=t):
                time_stamps = [index, index+alpha[0], index+alpha[0]+alpha[1], index+sum(alpha)]
                alpha_z_vectors.append([Y_train_norm[t_index] for t_index in time_stamps])
                #alpha_z_vectors.append(time_stamps)
                index = index+1
            z_vectors.append(alpha_z_vectors)
        return {'patterns': patterns, 'motifs': z_vectors}
    except Exception as e:
        print('Error while processing data: ' + str(e))
        return {'patterns': [], 'motifs': []}


def Unified_Prediction(predic_values, mode, weights):
    try:
        match mode:
            case 'avg':
                return sum(predic_values)/len(predic_values)
            case 'w_avg':
                return sum([weights[i]*value for i,value in enumerate(predic_values)])/sum(weights)
            case _:
                return None
    except Exception as e:
        print('Error while calculating a unified prediction value: ' + str(e))
        return None


def Prediction(data,t,h,eps,mode):
    predicted_values = {}
    unif_predic_values = {}
    try:
        Y_train, Y_test = data[:t+1], data[t+1:]
        Y_train_norm = Normalize(Y_train)
        Y_test_norm = Normalize(Y_test)
        dataSamples = Data_Samples(data,t,L=4,K_max=10)
        patterns, motifs = dataSamples['patterns'], dataSamples['motifs']
        for i in range(1, h+1):
            #Search for possible predicted values
            predicted_values[t+i] = []
            unif_predic_values[t+i] = None
            weights = []
            for j, alpha in enumerate(patterns):
                alpha_motifs = motifs[j]
                C = []
                index = t+i
                #From 1 to L-1
                for l in range(len(alpha)):
                    index = index - alpha[len(alpha)-1-l]
                    value = Y_train_norm[index] if index<t+1 else unif_predic_values[index]
                    if(value):
                        C.append(value)
                C.reverse()
                #Compare C to all motifs associated to alpha
                for motif in alpha_motifs:
                    C_alpha = motif
                    d = Eucl_Dist(C_alpha[:len(C_alpha)-1],C)
                    if(len(C)==len(C_alpha)-1 and d<eps):
                        predicted_values[t+i].append(C_alpha[len(C_alpha)-1])
                        weights.append((eps-d)/eps)
            #if t+i predictible, append it to Y_train_norm
            if(len(predicted_values[t+i])>0):
                value = Unified_Prediction(predicted_values[t+i], mode, weights)
                real_value = value*(max(Y_train)-min(Y_train))+min(Y_train)
                error = abs(real_value-Y_test[i-1])/abs(Y_test[i-1])
                print('mode {} error {}: {}'.format(mode, t+i, error))

                if (round(error,2)<=0.01):
                    unif_predic_values[t+i] = value
        return unif_predic_values
    except Exception as e:
        print('Error while predicting values: ' + str(e))
        return unif_predic_values


if __name__=="__main__":
    t = 1000
    h = 45
    eps = 0.001
    modes = ['avg', 'w_avg']
    print('Testing with BTC data from 01/01/2021 to 12/11/2023...')
    time_stamps, btcDailyHigh = Data_Import('data/bitcoin/bitcoin_2021-2023.csv')
    unif_predic_values_A = Prediction(btcDailyHigh,t,h,eps,mode='avg')
    unif_predic_values_B = Prediction(btcDailyHigh,t,h,eps,mode='w_avg')
    #Data visualization
    time = [i for i in range(t+h+1)]
    Y_real = btcDailyHigh[:t+h+1]
    Y_test_norm = Normalize(Y_real[t+1:])
    max_Y_train = max(btcDailyHigh[:t+1])
    min_Y_train = min(btcDailyHigh[:t+1])
    Y_pred_avg, Y_pred_wavg = [], []


    try:
        Y_pred_avg = Denormalize(unif_predic_values_A.values(), min=min_Y_train, max=max_Y_train)
        Y_pred_avg = np.array(Y_pred_avg)
        Y_pred_wavg = Denormalize(unif_predic_values_B.values(), min=min_Y_train, max=max_Y_train)
        Y_pred_wavg = np.array(Y_pred_wavg)
    except Exception as e:
        print('Unable to denormalize data: '+str(e))


    for i in range(1,h+1):
        if(t+i<len(btcDailyHigh)):
            print('Real value for {}: {}'.format(t+i, btcDailyHigh[t+i]))
        if Y_pred_avg[i-1]>0:
            print('Prediction with average: {}'.format(Y_pred_avg[i-1]))
            print('Prediction with weighted average: {}'.format(Y_pred_wavg[i-1]))
            print('----------------')
        else:
            print('point {} non predictable.'.format(t+i))
            print('----------------')

    #Charts
    plt.title('Estimation of BTC price high in USD')
    plt.grid()
    plt.xlabel('Time step')
    plt.ylabel('BTC value in USD')
    plt.plot(time[t+1:len(Y_real)], Y_real[t+1:], color='gray', label='Real values')
    pred_time = np.array(time[t+1:])
    plt.scatter(pred_time[Y_pred_avg>0], Y_pred_avg[Y_pred_avg>0], color='red', label='avg', marker=',', s=30)
    plt.scatter(pred_time[Y_pred_wavg>0], Y_pred_wavg[Y_pred_wavg>0], color='blue', label='w_avg', marker='^', s=30)
    plt.legend()
    plt.show()
