from matplotlib import pyplot as plt
noise = [0,0.05,0.1,0.15,0.2]
oat = [0.04802251459678246,0.04830069500039826,0.048576134582554005,0.04884875242129055,0.04911846437944632]
tnt = [0.09185840794634033,0.09215983460631605,0.09246013671192259,0.0927592929064781,0.09305728128975488]
tat = [0.01820935561851911,0.018847949353980768,0.01948654308944242,0.02011984062233296,0.0207479114919488]

ax = plt.gca() 
ax.plot(noise, oat,'c-o',label=r'$OAT$')
ax.plot(noise, tnt,'r-s',label=r'$TNT$')
ax.plot(noise, tat,'g-*',label=r'$TAT$')
ax.set_xlabel("noise")
ax.set_ylabel(r"$\xi^{2}_{S}$")
lgd = ax.legend(loc='center left', bbox_to_anchor=(0.8, 0.25))
plt.savefig("./noise.eps")