#Author : Danielle Azoulay
"""Genetic algorithm for cell-type composition estimation from gene expression data."""


import numpy as np
import random
import pandas as pd
from typing import List, Tuple



def genetic_algorithm(
        M: np.ndarray,
        H: np.ndarray,
        pop_size: int = 50,
        max_generations: int = 200,
        crossover_rate: float = 0.9,
        mutation_prob: float = 0.05,
        sigma: float = 0.1,
        tournament_size: int = 3,
        elite_fraction: float = 0.05,
        tol: float = 1e-10,
        patience: int = 30,
        use_non_uniform_mutation: bool = False,
        random_seed: int | None = None,
)-> Tuple[np.ndarray, float]:
    """
    Genetic Algorithm for RNA-seq deconvolution.
    Given Matrices M (genes X samples) and H (genes X cell types),
    searches for a matrix X (cell types X samples) such that:

    M ≈ H @ X

    The chromosome encodes X as a real-valued matrix, and each column of X
    represents valid cell-type proportions (non-negative, summing of 1).
    """
    #optional reproducibility:
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    #Dimensions: M is gXs, H is gXt -> X is tXs
    g, s = M.shape
    g2, t = H.shape
    if g != g2:
        raise ValueError("Gene dimension mismatch between M and H")

    # 1. Initialize population
    population = initialize_population(pop_size, t, s)
    fitnesses = [compute_fitness(ind, M, H) for ind in population]

    # Track best solution
    best_idx = int(np.argmax(fitnesses))
    best_X = population[best_idx].copy()
    best_fitness = fitnesses[best_idx]
    no_improve = 0

    generation = 0
    # 2. Main evolutionary loop (stop condition: max_generations or convergence)
    while generation < max_generations and no_improve < patience:
        generation += 1

        offspring: List[np.ndarray] = []
        offspring_fitnesses: List[float] = []

        #3. Parent selection + crossover + mutation -> offspring
        while len(offspring) < pop_size:
            # parent selection (tournament selection)
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)

            #crossover (intermediate recombination is the main operator)
            if random.random() < crossover_rate:
                child = crossover_intermediate(parent1, parent2)
            else:
                # No crossover: child is a copy of one parent
                child = parent1.copy()

            # Mutation (Gaussian, with optional non_uniform behavior)
            child = mutate_gaussian(
                child,
                mutation_prob,
                sigma,
                generation=generation,
                max_generations=max_generations,
                non_uniform=use_non_uniform_mutation,
            )


            # Evaluate new individual
            fit_child = compute_fitness(child, M, H)

            offspring.append(child)
            offspring_fitnesses.append(fit_child)

        # 4. Survivor selection (generational model with elitism)
        population, fitnesses = elitist_replacement(
            population,
            fitnesses,
            offspring,
            offspring_fitnesses,
            elite_fraction,
        )

        # 5. Update global best and check improvement
        current_best_idx = int(np.argmax(fitnesses))
        current_best_fit = fitnesses[current_best_idx]

        if current_best_fit > best_fitness + tol:
            best_fitness = current_best_fit
            best_X = population[current_best_idx].copy()
            no_improve = 0
        else:
            no_improve += 1
    # Ensure the final X is a valid proportion matrix
    best_X = normalize_X(best_X)
    return best_X, best_fitness


#Helper Functions


def normalize_X(X: np.ndarray) -> np.ndarray:
    """
    Normalize matrix X so that each column is a valid cell-type proportion vector:
    - All values are non-negative.
    - Each column sums to 1.
    if a column sums to zero, distribute mass uniformly across cell types.
    """
    # Enforce non-negativity (clip negative values)
    X = np.maximum(X,0.0)

    # Compute column sums
    col_sums = X.sum(axis=0, keepdims=True)
    
    # Handle columns that sum to zero (completely invalid)
    zero_cols_mask = (col_sums == 0.0)[0]
    if np.any(zero_cols_mask):
        X[:, zero_cols_mask] = 1.0 / X.shape[0]
        col_sums = X.sum(axis=0, keepdims=True)

    # Normalize each column to sum to 1
    return X / col_sums

def compute_fitness(X: np.ndarray, M:np.ndarray, H:np.ndarray) -> float:
    """
    Compute the fitness of candidate X:
    1. Normalize X to obtain valid cell-type fractions.
    2. Compute predicted expression M_hat = H @ X (vectorized matrix multiplication).
    3. Compute squared error between M and M_hat.
    4. Convert the error into a fitness value to be maximized.
    """
    X_norm = normalize_X(X)
    M_hat = H @ X_norm
    error = np.sum((M - M_hat) ** 2)
    fitness = 1.0 / (1.0 + error)
    return fitness

def initialize_population(pop_size: int, t: int, s: int) -> List[np.ndarray]:
    """
    Initialize a population of a random matrices X of size txs.
    Each X is normalized so that columns represent valid cell-type proportions.
    """
    population: List[np.ndarray] = []
    for _ in range(pop_size):
        X = np.random.rand(t, s)      # random non-negative values
        X = normalize_X(X)            # enforce valid proprtions
        population.append(X)
    return population

def tournament_selection(
        population: List[np.ndarray],
        fitnesses: List[float],
        tournament_size: int,
) -> np.ndarray:
    """
    Tournament selection:
    Randomly sample 'tournament size' individuals and return the fittest one.
    This induces selection pressure towards fitter individuals.
    """
    indices = random.sample(range(len(population)), tournament_size)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx].copy()

#crossover operators

def crossover_intermediate(
        parent1: np.ndarray,
        parent2: np.ndarray,
        alpha_low: float = 0.0,
        alpha_high: float = 1.0,
)-> np.ndarray:
    """
    Intermediate recombination for real-valued chromosomes:
    child = parent1 + alpha * (parent2 - parent1),
    where alpha is sampled independently for each gene.
    """
    alpha = np.random.uniform(alpha_low, alpha_high, size=parent1.shape)
    child = parent1 + alpha * (parent2 - parent1)
    return child

def crossover_line(
        parent1: np.ndarray,
        parent2: np.ndarray,
        alpha_low: float = 0.0,
        alpha_high: float = 1.0,
) -> np.ndarray:
    """
    Line recombination:
    A single alpha is drawn for the entire chromosome, creating a child on the line
    segment between the two parents.
    """
    alpha = random.uniform(alpha_low, alpha_high)
    child = parent1 + alpha * (parent2 - parent1)
    return child

def crossover_arithmetic(
        parent1: np.ndarray,
        parent2: np.ndarray,
        alpha: float = 0.5,
) -> np.ndarray:
    """
    Arithmetic crossover:
    child = alpha * parent1 + (1 - alpha) * parent2.
    This is another common operator for real coded GAs.
    """
    child = alpha * parent1 + (1-alpha) * parent2
    return child

#Mutation operators

def mutate_gaussian(
        individual: np.ndarray,
        mutation_prob: float,
        sigma: float,
        generation: int | None = None,
        max_generations: int | None = None,
        non_uniform: bool = False,
) -> np.ndarray:
    """
    Gaussian mutation:
    Adds small Gaussian noise to each gene with probability 'mutation_prob'.
    If non_uniform=True , sigma decreases over generations.
    (coarse search early, fine-tuning later).
    """
    t, s = individual.shape
    mutated = individual.copy()

    # Optionally decrease sigma over time (non_uniform nutation)
    if non_uniform and generation is not None and max_generations is not None:
        progress = generation / max_generations
        sigma_eff = sigma * (1.0 - progress)
        sigma_eff = max(sigma_eff , sigma * 0.1)    # Do not let sigma vanish completely
    else:
        sigma_eff = sigma

    for i in range(t):
        for j in range(s):
            if random.random() < mutation_prob:
                mutated[i, j] += random.gauss(0.0, sigma_eff)

    # Ensure the mutated individual remains legal
    mutated = normalize_X(mutated)
    return mutated

def mutate_random_reset(
        individual: np.ndarray,
        mutation_prob: float,
) -> np.ndarray:
    """
    Random-reseting mutation:
    Each Mutated gene receives a completely new random value in [0,1].
    After mutation, X is normalized to remain a valid solution.
    """
    t, s = individual.shape
    mutated = individual.copy()

    for i in range(t):
        for j in range(s):
            if random.random() < mutation_prob:
                mutated[i, j] = random.random()

    mutated = normalize_X(mutated)
    return mutated

#Survivor selection (replacement)

def elitist_replacement(
        population: List[np.ndarray],
        fitnesses: List[float],
        offspring: List[np.ndarray],
        offspring_fitnesses: List[float],
        elite_fraction: float,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Generational replacement with elitism:
    - Keep the top elite_fraction of the individuals from the current population.
    - Fill the rest of the population with offspring.
    This ensures that the best solutions are never lost.
    """

    pop_size = len(population)
    elite_size = max(1, int(elite_fraction *pop_size))

    # Indices of the best individuals in the current population
    sorted_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
    elite_indices = sorted_indices[:elite_size]

    new_population: List[np.ndarray] = [population[i].copy() for i in elite_indices]
    new_fitnesses: List[float] = [fitnesses[i] for i in elite_indices]

    # Fill the rest with offspring
    for ind, fit in zip(offspring, offspring_fitnesses):
        if len(new_population) >= pop_size:
            break
        new_population.append(ind)
        new_fitnesses.append(fit)
    return new_population, new_fitnesses


# main


if __name__ == "__main__":
    # Load input data
    M_path = "data/gene_sample_TPM_MatrixM.tsv"
    H_path = "data/gene_celltype_TPM_MatrixH.tsv"

    M_df = pd.read_csv(M_path, sep="\t", index_col=0)
    H_df = pd.read_csv(H_path,sep="\t", index_col=0)

    M = M_df.values.astype(float)
    H = H_df.values.astype(float)

    best_X, best_fit = genetic_algorithm(
        M,
        H,
        pop_size = 40,
        max_generations = 100,
        crossover_rate = 0.9,
        mutation_prob = 0.05,
        sigma = 0.1,
        tournament_size = 3,
        elite_fraction = 0.05,
        tol = 1e-10,
        patience = 30,
        use_non_uniform_mutation = False,
        random_seed = 42,
    )

    # Build a Dataframe for X with proper row/column labels
    X_df = pd.DataFrame(
        best_X,
        index=H_df.columns,    #celltype
        columns=M_df.columns   #samples
    )

    print("Best fitness:", best_fit)
    print("Estimated X matrix (cell-type proportions):")
    print(X_df)
    X_df.to_csv("results_X.tsv", sep="\t")















