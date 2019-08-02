#!/usr/bin/env python3

"""
Minimal character-level LSTM model. Written by Ngoc Quan Pham
Code structure borrowed from the Vanilla RNN model from Andreij Karparthy @karparthy.
BSD License
"""
import numpy as np
from random import uniform
import sys
import time

# Since numpy doesn't have a function for sigmoid
# We implement it manually here
def sigmoid(x):
  return 1 / (1 + np.exp(-x))


# The derivative of the sigmoid function
def dsigmoid(y):
    return y * (1 - y)


# The derivative of the tanh function
def dtanh(x):
    return 1 - x*x


# The numerically stable softmax implementation
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# data I/O
data = open('data/input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
std = 0.1

option = sys.argv[1]

# hyperparameters
emb_size = 8
hidden_size = 128  # size of hidden layer of neurons
seq_length = 32  # number of steps to unroll the RNN for
learning_rate = 10e-2
max_updates = 500000

print("Hyper Parameters:")
print("emb_size:", emb_size)
print("seq_length:", seq_length)
print("learning_rate:", learning_rate)
print("max_updates", max_updates)

concat_size = emb_size + hidden_size

# model parameters
# char embedding parameters
Wex = np.random.randn(emb_size, vocab_size)*std # embedding layer

# LSTM parameters
Wf = np.random.randn(hidden_size, concat_size) * std # forget gate
Wi = np.random.randn(hidden_size, concat_size) * std # input gate
Wo = np.random.randn(hidden_size, concat_size) * std # output gate
Wc = np.random.randn(hidden_size, concat_size) * std # c term

bf = np.zeros((hidden_size, 1)) # forget bias
bi = np.zeros((hidden_size, 1)) # input bias
bo = np.zeros((hidden_size, 1)) # output bias
bc = np.zeros((hidden_size, 1)) # memory bias

# Output layer parameters
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
by = np.zeros((vocab_size, 1)) # output bias


def forward(inputs, labels, memory):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """

    # The LSTM is different than the simple RNN that it has two memory cells
    # so here you need two different hidden layers
    hprev, cprev = memory

    # Here you should allocate some variables to store the activations during forward
    # One of them here is to store the hiddens and the cells
    hs, chats, cs, igates, ogates, fgates, ps, ys = {}, {}, {}, {}, {}, {}, {}, {}
    xs, wes, zs = {}, {}, {}

    hs[-1] = np.copy(hprev)
    cs[-1] = np.copy(cprev)

    loss = 0
    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
        xs[t][inputs[t]] = 1

        # convert word indices to word embeddings
        wes[t] = np.dot(Wex, xs[t])

        # LSTM cell operation
        # first concatenate the input and h
        # This step is irregular (to save the amount of matrix multiplication we have to do)
        # I will refer to this vector as [h X]
        zs[t] = np.row_stack((hs[t-1], wes[t]))

        # YOUR IMPLEMENTATION should begin from here

        # compute the forget gate
        # f_gate = sigmoid (W_f \cdot [h X] + b_f)
        fgates[t] = sigmoid(np.dot(Wf, zs[t]) + bf)
        

        # compute the input gate
        # i_gate = sigmoid (W_i \cdot [h X] + b_i)
        igates[t] = sigmoid(np.dot(Wi, zs[t]) + bi)

        # compute the candidate memory
        # \hat{c} = tanh (W_c \cdot [h X] + b_c])
        chats[t] = np.tanh(np.dot(Wc, zs[t]) + bc)

        # new memory: applying forget gate on the previous memory
        # and then adding the input gate on the candidate memory
        # c_new = f_gate * prev_c + i_gate * \hat{c}
        cs[t] = fgates[t] * cs[t-1] + igates[t] * chats[t]

        # output gate
        # o_gate = sigmoid (Wo \cdot [h X] + b_o)
        ogates[t] = sigmoid(np.dot(Wo, zs[t]) + bo)

        # new hidden state for the LSTM
        # h = o_gate * tanh(c_new)
        hs[t] = ogates[t] * np.tanh(cs[t])

        # DONE LSTM
        # output layer - softmax and cross-entropy loss
        # unnormalized log probabilities for next chars

        # o = Why \cdot h + by
        o = np.dot(Why, hs[t]) + by

        # softmax for probabilities for next chars
        # p = softmax(o)
        ps[t] = softmax(o)

        # cross-entropy loss
        # cross entropy loss at time t:
        # create an one hot vector for the label y
        ys[t] = np.zeros((vocab_size,1))
        ys[t][labels[t]] = 1

        # and then cross-entropy (see the elman-rnn file for the hint)
        loss_t = np.sum(-np.log(ps[t]) * ys[t])
        loss += loss_t

    # define your activations
    activations = (xs, zs, hs, chats, cs, fgates, ogates, igates, ps, ys)
    memory = (hs[len(inputs)-1], cs[len(inputs)-1])

    return loss, activations, memory


def backward(activations, clipping=True):
    xs, zs, hs, chats, cs, fgates, ogates, igates, ps, ys = activations

    # backward pass: compute gradients going backwards
    # Here we allocate memory for the gradients
    dWex, dWhy = np.zeros_like(Wex), np.zeros_like(Why)
    dby = np.zeros_like(by)
    dWf, dWi, dWc, dWo = np.zeros_like(Wf), np.zeros_like(Wi),np.zeros_like(Wc), np.zeros_like(Wo)
    dbf, dbi, dbc, dbo = np.zeros_like(bf), np.zeros_like(bi),np.zeros_like(bc), np.zeros_like(bo)

    # similar to the hidden states in the vanilla RNN
    # We need to initialize the gradients for these variables
    # these will not be returned
    dh = np.zeros_like(hs[0])
    dc = np.zeros_like(cs[0])
    dogatepre = np.zeros_like(ogates[0])
    dfgatepre = np.zeros_like(fgates[0])
    digatepre = np.zeros_like(igates[0])
   

    # back propagation through time starts here
    for t in reversed(range(len(inputs))):

        # IMPLEMENT YOUR BACKPROP HERE
        # refer to the file elman_rnn.py for more details
        do = ps[t] - ys[t]
        
        dWhy += np.dot(do, hs[t].T)
        dby += do

        dh = np.dot(Why.T, do) + dh

        dogatepre = dh * np.tanh(cs[t]) * dsigmoid(ogates[t])
        dWo += np.dot(dogatepre, zs[t].T)
        dbo += dogatepre
        
        dc = dh * ogates[t] * dtanh(np.tanh(cs[t])) + dc
        
        dc_hat_pre = dc * igates[t] * dtanh(chats[t])
        dWc += np.dot(dc_hat_pre, zs[t].T)
        dbc += dc_hat_pre

        digatepre = dc * chats[t] * dsigmoid(igates[t])
        dWi += np.dot(digatepre, zs[t].T)
        dbi += digatepre

        dfgatepre = dc * cs[t - 1] * dsigmoid(fgates[t])
        dWf += np.dot(dfgatepre, zs[t].T)
        dbf += dfgatepre

        dc = dc * fgates[t]

        dz = np.dot(Wc.T, dc_hat_pre) + np.dot(Wi.T, digatepre) + np.dot(Wf.T, dfgatepre) + np.dot(Wo.T, dogatepre)
        dh = dz[:hidden_size]
        dwes = dz[hidden_size:]
        dWex += np.dot(dwes, xs[t].T) 

    if clipping:
        # clip to mitigate exploding gradients
        for dparam in [dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby]:
            np.clip(dparam, -5, 5, out=dparam)

    gradients = (dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby)

    return gradients


def sample(memory, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    h, c = memory
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    generated_chars = []

    for t in range(n):
        # IMPLEMENT THE FORWARD FUNCTION ONE MORE TIME HERE
        # BUT YOU DON"T NEED TO STORE THE ACTIVATIONS
        e = np.dot(Wex, x)
        z = np.row_stack((h, e))

        in_gate = sigmoid(np.dot(Wi, z) + bi)
        f_gate = sigmoid(np.dot(Wf, z) + bf)
        c_hat = np.tanh(np.dot(Wc, z) + bc)
        c = c_hat * in_gate + c * f_gate
        o_gate = sigmoid(np.dot(Wo, z) + bo)
        h = o_gate * np.tanh(c)
        o = np.dot(Why, h) + by
        p = softmax(o)

        ix = np.random.multinomial(1, p.ravel())
        for j in range(len(ix)):
            if ix[j] == 1:
                index = j
        generated_chars.append(index)
        x = np.zeros((vocab_size, 1))
        x[index] = 1


    return generated_chars

if option == 'train':

    n, p = 0, 0
    n_updates = 0
    best = 1000000
    best_it = 0
    milestones = [150, 140, 130, 125, 120, 115, 110, 105, 100, 95, 92, 90, 88, 86, 84, 82, 80, 75, 70]
    iterations = [0] * len(milestones)
    milestonesTime = [0] * len(milestones)
    startTime = time.time()
    time100iterations = startTime

    # momentum variables for Adagrad
    mWex, mWhy = np.zeros_like(Wex), np.zeros_like(Why)
    mby = np.zeros_like(by) 

    mWf, mWi, mWo, mWc = np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wo), np.zeros_like(Wc)
    mbf, mbi, mbo, mbc = np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bo), np.zeros_like(bc)

    smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
    
    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p+seq_length+1 >= len(data) or n == 0:
            hprev = np.zeros((hidden_size,1)) # reset RNN memory
            cprev = np.zeros((hidden_size,1))
            p = 0 # go from start of data
        
        inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
        targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

        # sample from the model now and then
        if n % 100 == 0:
            sample_ix = sample((hprev, cprev), inputs[0], 2000)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print ('----\n %s \n----' % (txt, ))


        # forward seq_length characters through the net and fetch gradient
        loss, activations, memory = forward(inputs, targets, (hprev, cprev))
        gradients = backward(activations)

        hprev, cprev = memory
        dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients
        smooth_loss = smooth_loss * 0.999 + loss * 0.001 
        if n % 100 == 0: 
            smooth_lossIF = smooth_loss * (64 / seq_length)
            if smooth_lossIF < best:
                best = smooth_lossIF
                best_it = n
            now = time.time()
            for i in range(len(milestones)):
                if smooth_lossIF <= milestones[i] and iterations[i] == 0: 
                    iterations[i] = n
                    milestonesTime[i] = now - startTime
            print ('iter %d, loss: %.2f, best: %.2f (iter %d), time for 100 iterations: %d, runtime: %d' % (n, smooth_lossIF, best, best_it, (now - time100iterations), (now - startTime))) # print progress
            for i in range(len(iterations)):
                if iterations[i] != 0:
                    print('milestone %d in %d iteraions (%d seconds)' % (milestones[i], iterations[i], milestonesTime[i]))
            time100iterations = time.time()

            sys.stdout.flush()

        # perform parameter update with Adagrad
        for param, dparam, mem, name in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by],
                                    [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex, dWhy, dby],
                                    [mWf, mWi, mWo, mWc, mbf, mbi, mbo, mbc, mWex, mWhy, mby], ["Wf","Wi","Wo","Wc","bf","bi","bo","bc","Wex","Why","by"]):
#            print("  ",name, param.shape, dparam.shape)
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

        p += seq_length # move data pointer
        n += 1 # iteration counter
        n_updates += 1
        if n_updates >= max_updates:
            break

elif option == 'gradcheck':

    p = 0
    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    delta = 0.001

    hprev = np.zeros((hidden_size, 1))
    cprev = np.zeros((hidden_size, 1))

    memory = (hprev, cprev)

    loss, activations, _ = forward(inputs, targets, memory)
    gradients = backward(activations, clipping=False)
    dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients

    for weight, grad, name in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by], 
                                   [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex    , dWhy, dby],
                                   ['Wf', 'Wi', 'Wo', 'Wc', 'bf', 'bi', 'bo', 'bc', 'Wex', 'Why', 'by']):

        str_ = ("Dimensions dont match between weight and gradient %s and %s." % (weight.shape, grad.shape))
        assert(weight.shape == grad.shape), str_

        print(name)
        for i in range(weight.size):
      
            # evaluate cost at [x + delta] and [x - delta]
            w = weight.flat[i]
            weight.flat[i] = w + delta
            loss_positive, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w - delta
            loss_negative, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w  # reset old value for this parameter

            grad_analytic = grad.flat[i]
            grad_numerical = (loss_positive - loss_negative) / ( 2 * delta )

            # compare the relative error between analytical and numerical gradients
            abs_a = abs(grad_analytic - grad_numerical)
            abs_b = abs(grad_numerical + grad_analytic)
            try:
                rel_error = abs_a / abs_b
            except:
                print(f"a:{abs_a}, b:{abs_b}")

            # rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)

            if rel_error > 0.01:
                print ('WARNING %f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
