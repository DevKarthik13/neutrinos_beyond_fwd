#!/usr/bin/env python

import numpy as np
import sys
from itertools import combinations

filename = sys.argv[1]
pmax = int(sys.argv[2])

dp = 1.0
pmin = -pmax
np1 = round((pmax-pmin)/dp) + 1

ps_grid = np.zeros((np1**2,3))
pxs, pys = np.meshgrid(np.linspace(pmin,pmax,np1), np.linspace(pmin,pmax,np1))
ps_grid[:,0] = pxs.reshape(np1**2)
ps_grid[:,1] = pys.reshape(np1**2)
ps_grid[:,2] = np.zeros(np1**2)

kes_grid = np.zeros(np1**2)
kes_grid = np.sqrt(ps_grid[:,0]**2+ps_grid[:,1]**2+ps_grid[:,2]**2)

ps = []
kes = []

# Impose ke < pmax
for i in range(len(kes_grid)):
    if kes_grid[i] < pmax+1e-5 and kes_grid[i] > 0 and ps_grid[i,0] >  0:
        ps.append(ps_grid[i])
        kes.append(kes_grid[i])

np2 = len(kes)
ps = np.array(ps)
kes = np.array(kes)

print(f'Number of momentum modes: {np2}')
print('Momentum modes:')
for i in range(np2):
    print(i, ps[i,:])
        
kec_nt = []

# Input is 3-dim momentum, returns |p|, theta and phi (0-2pi)
def p_spherical(p):
    absp = np.sqrt(np.sum(p*p))
    theta = np.arctan2(np.sqrt(p[0]**2+p[1]**2), p[2])
    phi = np.arctan2(p[1],p[0])
    return absp, theta, phi

# g factor for H_vv
def g_factor(p1, p2, q1, q2):
    absp1, tp1, pp1 = p_spherical(p1)
    absp2, tp2, pp2 = p_spherical(p2)
    absq1, tq1, pq1 = p_spherical(q1)
    absq2, tq2, pq2 = p_spherical(q2)
    fac1 = np.exp(-1j*pq1)*np.sin(tq1/2.)*np.cos(tq2/2.) - np.exp(-1j*pq2)*np.cos(tq1/2.)*np.sin(tq2/2.)
    fac2 = np.exp(1j*pp1)*np.sin(tp1/2.)*np.cos(tp2/2.) - np.exp(1j*pp2)*np.cos(tp1/2.)*np.sin(tp2/2.)
    return 2 * fac1 * fac2

for i1 in range(np2):
    p1 = ps[i1,:]
    for i2 in range(i1,np2):
        p2 = ps[i2,:]
        for i3 in range(np2):
            p3 = ps[i3,:]
            for i4 in range(i3,np2):
                p4 = ps[i4,:]
                pdiff = p1+p2-p3-p4
                if np.sum(np.abs(pdiff)) < 1e-7:
                    k1 = np.sqrt(np.dot(p1,p1))
                    k2 = np.sqrt(np.dot(p2,p2))
                    k3 = np.sqrt(np.dot(p3,p3))
                    k4 = np.sqrt(np.dot(p4,p4))
                    gf = g_factor(p1,p2,p3,p4)
                    if np.abs(k1+k2-k3-k4) < 1e-7 and np.abs(gf) > 1e-7:
                        if i1!=i3:
                            kec_nt.append([i1,i2,i3,i4])


kec_nt = np.array(kec_nt)
Nkec = len(kec_nt)

# reference state
ninit = np.sort(np.array([0,5,8,10,12,20,25,26,28,33]))
Nn = len(ninit)

pinit = np.sum(ps[ninit], axis=0)
keinit = np.sum(kes[ninit])
print(f'Reference state is {ninit}')
print(f'Reference state total momentum: {pinit}')
print(f'Reference state kinetic energy: {keinit}')

# Write momentum modes and other initial settings to 'filename'_p.dat file
g =  open(filename+'_p.dat', 'w')
g.write(f'# dp = {dp}' + '\n')
g.write(f'# pmax = {pmax}' + '\n')
g.write(f'# Initial state is {ninit}' + '\n')
g.write(f'# Initial total momentum: {pinit}' + '\n')
g.write(f'# Initial kinetic energy: {keinit}' + '\n')
g.write(f'# Number of momentum modes: {np2}' + '\n')
# Write pairs of momentum modes which conserve P and KE
for i in range(len(kec_nt)):
    gfac = g_factor(ps[kec_nt[i,0],:],ps[kec_nt[i,1],:],ps[kec_nt[i,2],:],ps[kec_nt[i,3],:])
    g.write(f'# {kec_nt[i,0]} {kec_nt[i,1]} --- {kec_nt[i,2]} {kec_nt[i,3]}: {np.abs(gfac)}' + '\n')
# Write momentum modes
for i in range(len(ps)):
    g.write(f'{ps[i,0]} {ps[i,1]} {ps[i,2]}' + '\n')


binom = np.array(list(combinations([i for i in range(Nn)],2)))
Nbinom = len(binom)

# Check if the same momentum mode is used only up to twice
def check(state):
    ans = True
    for i in range(len(state)-2):
        if np.var(state[i:i+3]) < 1e-7:
            ans = False
    return ans

# For a given pair of Nn momentum modes, return the pairs that Hvv can take to
def apply(state):
    newstate = []
    for i in range(Nbinom):
        k = np.array([state[binom[i,0]], state[binom[i,1]]])
        for j in range(Nkec):
            if np.sum(np.abs(k-kec_nt[j,:2]))==0:
                state_i = state.copy()
                state_i[binom[i,0]] = kec_nt[j,2]
                state_i[binom[i,1]] = kec_nt[j,3]
                state_i = np.sort(state_i)
                if check(state_i):
                    newstate.append(state_i)
    return np.array(newstate)

nnewstate = 10  # > 1 to enter the while loop
p_states = np.array([ninit])
newstate1 = np.array([np.zeros(Nn), ninit])
trial = 0
while nnewstate > 1:
    print(f'---------------- H^{trial+1} ----------------')
    newstate2 = np.zeros((1,Nn))
    #newstate3 = np.zeros((1,Nn))
    for i in range(1,len(newstate1)):
        newi = apply(newstate1[i])
        if len(newi) > 0:
            newstate2 = np.append(newstate2, newi, axis=0)
       
    newstate1 = np.array([np.zeros(Nn)])
    for j in range(1,len(newstate2)):
        dist = np.sum(np.abs(p_states - newstate2[j]), axis=1)
        if np.min(dist) > 1e-7:
            p_states = np.append(p_states, [newstate2[j].astype(int)], axis=0)
            newstate1 = np.append(newstate1, [newstate2[j]], axis=0)
    nnewstate = len(newstate1)
    print(f'Number of new states(mod flavor choice) at this round: {nnewstate-1}')
    print(f'Number of states(mod flavor choice) visited so far: {len(p_states)}')
        
    trial += 1

print('#########################')
print('Finished listing all basis states (mod flavor choice) connected to the reference state')

n_states = 0

# Write all basis states visited (mod flavor choice) to 'filename'_s.dat file.
with open(filename + '_s.dat', 'w') as f:    
    for i in range(len(p_states)):
        totp = np.sum(ps[p_states[i]], axis=0)
        totke = np.sum(kes[p_states[i]])
        nstate = 2**Nn/4**(Nn-len(set(p_states[i])))
        strs = ' '.join([str(x) for x in p_states[i]]) + '\n'
        f.write(strs)
        n_states += nstate
        if np.sum(np.abs(totp-pinit)) > 1e-7 or np.abs(totke-keinit) > 1e-7:
            print('Total momentum or kinetic energy is not conserved!')


print(f'Number of momentum modes pair with conserved P and E is {len(p_states)}')
print(f'Number of states with conserved P,E, and arbitrary flavor contents is {n_states:.0f}')
bins_visited = np.sort(np.array(list(set(p_states.flatten()))))
print(f'Activated momentum modes:')
print(bins_visited)


g.write(f'# bins visited: {bins_visited}'+'\n')
g.write(f'# Number of basis states with conserved P,E, and arbitrary flavor contents is {n_states:.0f}')
g.close()

exit()


