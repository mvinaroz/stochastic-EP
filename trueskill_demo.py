import numpy as np 
from scipy.stats import norm

def compute_functions(mean, var):

    value_arg=np.divide(mean, np.sqrt(var))

    pdf_normal=norm.pdf(value_arg)
    cdf_normal=norm.cdf(value_arg)
    psi_func=np.divide(pdf_normal, cdf_normal)

    sum_term=value_arg + psi_func
    lambda_func=np.multiply(psi_func, sum_term)

    return psi_func, lambda_func


def trueskillEP(num_players, data, num_iter):
    '''
    num_players: total number of payers.
    data[i, 0]: id for the winner
    data[i, 1]: id for the loser
    '''
    num_games=data.shape[0]

    pv=0.5 #Prior skill variance (prior mean is always 0)

    #Initialize skill marginals for each player (mean and variance).
    Ms=np.empty(num_players)
    #Ms[:]=np.nan
    Ps=np.empty(num_players)
    #Ps[:]=np.nan

    #Initialize upward messages from game factor h_g to s_i variables (mean and prcision). (THESE ARE THE FACTORS IN EP f_n(\theta))
    Mgs=np.zeros((num_games, 2))
    Pgs=np.zeros((num_games, 2))

    #Initialize matrices of skills to game messags (mean and precision). (THESE ARE THE CAVITY DISTRIBUTIONS q_{-n}(\theta))
    Msg=np.zeros((num_games, 2))
    Psg=np.zeros((num_games, 2))
    #print("Msg before update: ", Msg)

    for iter in range(num_iter):

        #Step 1. Compute the posterior over skills.
        for player in range(num_players):
            Ps[player]=1/pv + np.sum(Pgs[np.isin(data, player+1)])
            Ms[player]=np.dot(Pgs[np.isin(data, player+1)], Mgs[np.isin(data, player+1)])/Ps[player]
            #print(Pgs[np.isin(data, i+1)])
            #print(Mgs[np.isin(data, i+1)])

        print("This is Ps: ", Ps)
        print("This is Ms: ", Ms)
        #Step 2. Compute skill to game messages (i.e cavity distribution with respect to a game and its palyers).
        data_index=data - 1 #Need to select elements in Ps by index of player in Ps.
        #print(data_index)
        D=data_index.reshape(-1)

        Ps_per_game_played=np.take(Ps, D).reshape(data.shape)
        Ms_per_game_played=np.take(Ms, D).reshape(data.shape)

        Psg=Ps_per_game_played - Pgs
        #print("Psg: ", Psg)
        term1_msg=np.multiply(Ps_per_game_played, Ms_per_game_played) - np.multiply(Pgs, Mgs)
        Msg=np.divide(term1_msg, Psg)
        print(Msg)

        #Step 3. Compute game to performance messages. 
        vgt=1+ np.sum(1/Psg, axis=1)
        mgt=Msg[:,0] - Msg[:, 1] #Player in the first column  always beats the player on the second column.

        #Step 4. Approximate the marginal on performance differences.
        psi_func, lambda_func=compute_functions(mgt, vgt)
        Mt=mgt + np.multiply(np.sqrt(vgt), psi_func)
        Pt=1/np.multiply(vgt, 1-lambda_func)

        #Step 5. Compute performance to game messages.
        ptg=Pt - 1/vgt
        mtg_term1=np.multiply(Mt, Pt) - np.multiply(mgt, vgt)
        mtg=np.divide(mtg_term1, ptg)


        #Step 6. Compute gqme to skill messages ( f_{n}(\theta) updates for players in game i)
        ptg_inv=(1/ptg).reshape(-1,1)
        ptg_repeat=np.concatenate((ptg_inv, ptg_inv) , axis=1)
        Psg_reverse=1/np.flip(Psg, axis=1)
        pgs_denom=1+ptg_repeat+Psg_reverse
        
        Pgs=1/pgs_denom

        mtg_repeat=np.concatenate((mtg.reshape(-1,1), -mtg.reshape(-1,1)) , axis=1)

        Mgs= mtg_repeat + np.flip(Msg, axis=1)

    return Ms, Ps
    




# Let us assume the following partial order on players
# where higher up is better
#     1
#    /  \
#    2   3
#     \/
#     4
#    /  \
#   5    6
# We will sample data from this graph, where we let each player
# beat its children K times

Nplayers=6
G=np.zeros((Nplayers, Nplayers))
G[0, 1]=1
G[0, 2]=1
G[1, 3]=1
G[2, 3]=1
G[3, 4]=1
G[3, 5]=1

#print(G)

K=np.random.randint(1, high=5, size=6, dtype=int) #Generate the number of games between players 1-2, 1-3, 2-4, 3-4, 4-5, 4-6
#print(K)
#data=np.zeros((np.sum(K), 2))
list_players=[[1,2], [1, 3], [2,4], [3,4], [4,5], [4, 6]]

#Generate data where column 1 is the winner and column 2 is the loser of each game.
for k in range(K.shape[0]):
    data12=np.full((K[k], 2), list_players[k])
    if k==0:
        data=data12
    else:
        data=np.concatenate((data, data12), axis=0)

print(data)

num_iter=5
Ms, Ps=trueskillEP(Nplayers, data, num_iter)
