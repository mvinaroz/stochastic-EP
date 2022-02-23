from re import A
import numpy as np 
from scipy.stats import norm

def trueskillEP(num_players, data, num_iter):
    '''
    num_players: total number of payers.
    data[i, 0]: id for the winner
    data[i, 1]: id for the loser
    '''
    num_games=data.shape[0]

    pv=0.5 #Prior skill variance (prior mean is always 0)

    #Prior skill variance and mean.
    psi_var=0.5
    psi_mean=0.0

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
    print("Msg before update: ", Msg)

    for iter in range(num_iter):

        for i in range(num_games):

            #Step 1. Compute the posterior over skills.
            for player in range(num_players):
                Ps[player]=1/pv + np.sum(Pgs[np.isin(data, player+1)])
                Ms[player]=np.dot(Pgs[np.isin(data, player+1)], Mgs[np.isin(data, player+1)])/Ps[player]
                #print(Pgs[np.isin(data, i+1)])
                #print(Mgs[np.isin(data, i+1)])
        
            print("this is Ps: ", Ps)

            game_played=data[i, :] - 1 #index of the i-th data row (game played)
            #Step 2. Compute skill to game messages (i.e cavity distribution with respect to a game and its palyers).
            print("The game played: ", game_played)
            Psg[i, :]=Ps[game_played] - Pgs[i, :]
            Msg[i, :]=(np.dot(Ps[game_played], Ms[game_played]) - np.dot(Pgs[i], Mgs[i])) / Psg[i,:]

            #Step 3. Compute game to performance messages. 
            vgt= 1+ np.sum(1./Psg[i,:])
            mgt=Msg[i, 0] - Msg[i, 1] #The first player always wins the second one.

            #Step 4. Approximate the marginal on performance differences.
            funcs_arg=mgt /np.sqrt(vgt)
            lambda_func=norm.pdf(funcs_arg,loc=0,scale=1)*(norm.pdf(funcs_arg,loc=0,scale=1) + funcs_arg)
            Mt=mgt + np.sqrt(vgt)*norm.pdf(funcs_arg,loc=0,scale=1)
            Pt= 1./(vgt*(1 - lambda_func) )

            #Step 5. Compute performance to game messages.
            ptg=Pt  - 1./vgt
            mtg= (Mt*Pt - mgt/vgt)/ptg

            #Step 6. Compute gqme to skill messages ( f_{n}(\theta) updates for players in game i)
            Pgs[i, :]= 1+ ptg + Psg[i,:][::-1] 
            Mgs[i, 0]= mtg + Msg[i, 1]
            Mgs[i, 1]=mtg - Msg[i, 0]




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

num_iter=1
trueskillEP(Nplayers, data, num_iter)
