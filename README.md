# But du projet

Dans ce projet, il est question de trouver un optimum (ici le minimum) de la fonction de Easom en utilisant la méthode d'optimisation par essaims de particules (Particle Swarm Optimisation ou PSO).

La fonction de Easom est définie par:

$$ f(x,y) = -\cos(x)\cos(y)\exp(-((x - \pi)^2 + (y - \pi)^2)) $$ 

avec :

$x, y \in [-100, 100]$

Son minimum global est : $f(X^*) = -1$ en $X^* = (\pi,\pi)$, C'est ce qu'on va chercher à approcher avec la méthode PSO.



# PSO c'est quoi ?


