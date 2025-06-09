#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def viterbi(observations, states, start_prob, trans_prob, emit_prob):
    """
    Applica l'algoritmo di Viterbi per un HMM discreto:
    - observations: lista di symboli osservati (es. ['walk', 'shop', ...])
    - states: lista di stati nascosti (es. ['Sunny', 'Rainy'])
    - start_prob[state]: probabilità iniziale π_i
    - trans_prob[i][j]: probabilità di transizione a_{ij}
    - emit_prob[state][obs]: probabilità di emissione b_i(o)
    
    Ritorna:
    - path: sequenza di stati (nomi) che massimizza la probabilità
    - delta: matrice (T × N) con le probabilità massime per ogni passo t e stato i
    """
    T = len(observations)    # lunghezza della sequenza osservata
    N = len(states)          # numero di stati nell'HMM
    delta = np.zeros((T, N)) # δ[t, i]: massima prob. di sequenza fino a t, finendo in stato i
    psi = np.zeros((T, N), dtype=int)  # per tracciare il predecessore ottimo

    # ------ Inizializzazione t = 0 ------
    for i, s in enumerate(states):
        # δ[0,i] = π_i * b_i(o_0)
        delta[0, i] = start_prob[s] * emit_prob[s].get(observations[0], 0.0)

    # ------ Ricorrenza t = 1..T-1 ------
    for t in range(1, T):
        for j, sj in enumerate(states):
            # calcolo la probabilità massima proveniente da tutti i possibili stati precedenti
            probs = [
                delta[t-1, i] *
                trans_prob[si].get(sj, 0.0) *     # a_{i→j}
                emit_prob[sj].get(observations[t], 0.0)  # b_j(o_t)
                for i, si in enumerate(states)
            ]
            delta[t, j] = max(probs)               # memorizzo il valore massimo
            psi[t, j] = int(np.argmax(probs))      # e da quale stato viene

    # ------ Terminazione + Ricostruzione percorso ------
    path_idx = np.zeros(T, dtype=int)
    path_idx[T-1] = int(np.argmax(delta[T-1]))  # ultimo stato massimo
    for t in range(T-2, -1, -1):                # risalgo con psi
        path_idx[t] = psi[t+1, path_idx[t+1]]
    path = [states[i] for i in path_idx]       # converto indici in nomi stati

    return path, delta

# ------ MAIN: settaggio HMM e plot ------
if __name__ == "__main__":
    # Definizione modello a due stati
    states = ['Sunny', 'Rainy']
    observations = ['walk', 'shop', 'clean', 'walk', 'walk']
    start_prob = {'Sunny': 0.6, 'Rainy': 0.4}
    trans_prob = {
        'Sunny': {'Sunny': 0.7, 'Rainy': 0.3},
        'Rainy': {'Sunny': 0.4, 'Rainy': 0.6}
    }
    emit_prob = {
        'Sunny': {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
        'Rainy': {'walk': 0.1, 'shop': 0.4, 'clean': 0.5}
    }

    # Eseguo Viterbi
    path, delta = viterbi(observations, states, start_prob, trans_prob, emit_prob)

    # ------ Plot 1: heatmap delle probabilità δ(t,i) ------
    plt.figure()
    plt.imshow(delta, aspect='auto', interpolation='nearest')
    plt.colorbar(label='δ(t,i) – massima prob.')
    plt.xlabel('Stato (0=Sunny, 1=Rainy)')
    plt.ylabel('Tempo t')
    plt.title('Probabilità massime δ(t,i)')
    plt.tight_layout()

    # ------ Plot 2: sequenza di stati ottima ------
    plt.figure()
    plt.step(range(len(path)), [states.index(s) for s in path], where='post')
    plt.yticks([0,1], states)
    plt.xlabel('Tempo t')
    plt.ylabel('Stato nascosto')
    plt.title('Percorso Viterbi ottimo')
    plt.tight_layout()

    # ------ Plot 3: evoluzione di max δ(t) ------
    plt.figure()
    plt.plot(np.max(delta, axis=1), marker='o')
    plt.xlabel('Tempo t')
    plt.ylabel('max δ(t)')
    plt.title('Evoluzione della probabilità massima')
    plt.tight_layout()

    # Visualizzo tutti i grafici
    plt.show()

    # Stampa del percorso finale
    print("Percorso Viterbi:", path)
