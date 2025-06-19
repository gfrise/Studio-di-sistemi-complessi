import networkx as nx #Richiamo la libreria per la gestione dei network
import random
import EoN #Richiamo Epidemics on Network, libreria per la simulazione di epidemie su network

G=nx.Graph() #Creo il grafico su cui costruisco
population=250

i=0
while i<population: #Creazione popolazione
    G.add_nodes_from([str(i)])
    i=i+1

i=0
while i<population: #Creazione casuale dei collegamenti tra punti
    a=random.randrange(0,population)
    b=random.randrange(0,population)
    if i != b:
        G.add_edge(str(i),str(b))
    if i != a:
        G.add_edge(str(i),str(a))
    i=i+1

beta = 1 #tasso di contaggio/velocità di tramissione
gamma = 0.2 #tasso di recupero dei contaggiati
r_0 = beta/gamma #numero di riproduzione base della malattia

print("Valore riproduzione base della malattia:"+str(r_0))


#I infetti, R rimessi, S suscettibili
I0 = 1
R0 = 0
S0 = population-I0-R0

pos = nx.spring_layout(G)
          
print("Simulo l'andamento dell'epidemia attraverso l'algoritmo Gillespie")
sim = EoN.Gillespie_SIR(G, tau=beta, gamma=gamma, rho = I0/population, return_full_data=True)
print("Simulazione avvenuta con successo")
print("Stampa dei grafici")
for i in range (0,15,1):
    sim.display(time = i)