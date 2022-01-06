# make sure to install autodp in your environment by
# pip install autodp

from autodp import rdp_acct, rdp_bank

# get the CGF functions
def CGF_func(sigma1, sigma2):

  func_gaussian_1 = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma1}, x)
  func_gaussian_2 = lambda x: rdp_bank.RDP_gaussian({'sigma': sigma2}, x)

  func = lambda x: func_gaussian_1(x) + func_gaussian_2(x)

  return func


def main():

    """ input arguments """

    # (1) privacy parameters for four types of Gaussian mechanisms
    sigma1 = 2. # noise level for privatizing mean
    sigma2 = 2. # noise level for privatising covariance

    # (2) desired delta level
    delta = 1e-5

    # (3) sampling rate
    n_epochs = 20  # depending on your experiment length, change the number of epochs for training
    batch_size = 500  # depending on your mini-batch size, change this value

    n_data = 50000  # depending on your dataset size, change this value
    steps_per_epoch = n_data // batch_size
    k = steps_per_epoch * n_epochs # k is the number of steps during the entire training
    prob = batch_size / n_data # prob is the subsampling probability

    """ end of input arguments """

    """ now use autodp to calculate the cumulative privacy loss """
    # declare the moment accountants
    acct = rdp_acct.anaRDPacct()

    # define the functional form of uppder bound of RDP
    func = CGF_func(sigma1, sigma2) # we redefine CFG for double Gaussian mechanisms

    eps_seq = []
    print_every_n = 100
    for i in range(1, k + 1):
      acct.compose_subsampled_mechanism(func, prob)
      eps_seq.append(acct.get_eps(delta))
      if i % print_every_n == 0 or i == k:
        print("[", i, "]Privacy loss is", (eps_seq[-1]))

    print("The final epsilon delta values after the training is over: ", (acct.get_eps(delta), delta))


if __name__ == '__main__':
  main()
