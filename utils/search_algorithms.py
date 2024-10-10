import random
import math
import numpy as np
from typing import Tuple, List, Callable, Optional, Dict, Union, Any
import matplotlib.pyplot as plt
from math import floor

#This is how we discretize the intervals
# This is PSO using Qlearning


def discretize_state(distance: Tuple[float], discrete_interval_legth: float, num_states: int) -> Tuple[int]:
    best_this_iter = int(distance[0] // discrete_interval_legth)
    best_global = int(distance[1] // discrete_interval_legth)
    if best_this_iter > num_states:
        best_this_iter = num_states
    if best_global > num_states:
        best_global = num_states
    return (best_this_iter, best_global)

class QLearningTable:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.4, load: bool = False, index: int = 0):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.num_states = num_states
        self.num_actions = num_actions
        rows = self.num_states[0]
        cols = self.num_states[1]

        # Initialize Q-table with zeros
        if load:
            self.q_table = np.load(f'particle_data/p_{index}.npy')
        else:
            self.q_table = np.random.uniform(low=0, high=1, size=(rows, cols, num_actions))


    def select_action(self, state):
        # Epsilon-greedy policy to select action
        if np.random.rand() < self.exploration_prob:
            # Exploration: Choose a random action
            return np.random.randint(self.num_actions)
        else:
            # Exploitation: Choose action with the highest Q-value
            return np.argmax(self.q_table[state])

    def update_q_value(self, state, action, reward, next_state):
        # Q-value update based on the Bellman equation
        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state, action] = new_q


##############################################################################################################
############ Local Search (Hill Climbing) Algorithm ##########################################################
##############################################################################################################
def local_search(cost_function: Callable, max_itr: int, convergence_threshold: float, 
                 x_initial: Optional[np.array] = None, x_range: Optional[List[List[float]]] = None, hide_progress_bar: Optional[bool] = False) -> Tuple[np.array, float, List[np.array], List[float]]:
    # Set the x_initial
    if x_initial is None:
        x_initial = [random.uniform(x_range[i][0], x_range[i][1]) for i in range(len(x_range))]

    x_current = x_initial
    cost_current = cost_function(x_current)

    x_history = [x_current]
    cost_history = [cost_current]

    convergence = False
    itr = 0
    while not convergence:
        # Generate neighboring solutions
        x_neighbor = [random.gauss(x, 0.1) for x in x_current]
        x_neighbor = bound_solution_in_x_range(x=x_neighbor, x_range=x_range)
        cost_neighbor = cost_function(x_neighbor)

        # Accept the neighbor if it has lower cost
        if cost_neighbor < cost_current:
            x_current = x_neighbor
            cost_current = cost_neighbor
            if (cost_current < convergence_threshold) or (itr >= max_itr):
                convergence = True
        
        if abs(cost_current-cost_neighbor) < convergence_threshold:
            break

        x_history.append(x_current)
        cost_history.append(cost_current)

        itr += 1
    

    # Get the best solution
    best_cost_index = np.argmin(cost_history)
    best_x = x_history[best_cost_index]
    best_cost = cost_history[best_cost_index]

    return best_x, best_cost, x_history, cost_history

##############################################################################################################
############ Particle Swarm Optimization (PSO) ###############################################################
##############################################################################################################
def qlearning_pso(cost_function: Callable, num_particles: int, max_itr: int, alpha_1: float, alpha_2: float, alpha_3: float,
        x_initial: Optional[np.array] = None, x_range: Optional[List[List[float]]] = None,
        local_best_option: Optional[str] = 'this_iteration', global_best_option: Optional[str] = 'this_iteration',
        ls_max_itr: Optional[int] = 100, ls_convergence_threshold: Optional[float] = 0.01) -> Tuple[np.array, float, List[np.array], List[float]]:
    
    NUM_STATES = 1000
    d = len(x_range)
    max_in_range = x_range[0][1]
    distance_normalizer = np.linalg.norm(x=np.array(object=[max_in_range for _ in range(d)]))
    discrete_interval_legth = distance_normalizer / NUM_STATES
    # Set the x_initial
    if x_initial is None:
        x_initial = [random.uniform(x_range[i][0], x_range[i][1]) for i in range(len(x_range))]
    
    # Initialize particles (candidate solutions)
    particles = [{'position': np.array([random.uniform(x[0], x[1]) for x in x_range]),
                  "prev_cost": float('inf'),
                  "curr_cost_for_position": None,
                  'velocity': np.array([random.uniform(-1, 1) for _ in range(len(x_range))]),
                  'best_position': x_initial,
                  'best_cost': float('inf'),
                  'position_history': [],
                  "q_table": QLearningTable(num_actions=3, num_states=(NUM_STATES+1, NUM_STATES+1), load=False, index=i),
                  "curr_state": tuple(np.random.randint(low=0, high=100, size=2)),
                  "new_state": None,
                  "phi": 2.2,
                } for i in range(num_particles)]
    
    for particle_index in range(len(particles)):
        particles[particle_index]["curr_action"] = particles[particle_index]["q_table"].select_action(particles[particle_index]["curr_state"])
    
    # Initialize global best
    global_best_position = x_initial
    global_best_cost = float('inf')
    
    x_history = []
    cost_history = []

    for iter in range(max_itr):
        best_xs_in_this_iteration, best_costs_in_this_iteration = [], []
        
        for particle_index in range(len(particles)):
            # Do local search (every particle searches locally in the local neighborhood)
            best_x, best_cost, _, _ = local_search(cost_function=cost_function, max_itr=ls_max_itr, convergence_threshold=ls_convergence_threshold,
                                                   x_initial=particles[particle_index]['position'], x_range=x_range, hide_progress_bar=True)

            particles[particle_index]["curr_cost_for_position"] = best_cost

            # Find the local best particle (for use in the velocity vector):

            if local_best_option == 'this_iteration':
                local_best_x = best_x
            elif local_best_option == 'so_far':
                if best_cost < particles[particle_index]['best_cost']:
                    particles[particle_index]['best_cost'] = best_cost
                    particles[particle_index]['best_position'] = best_x
                    local_best_x = particles[particle_index]['best_position']
            
            best_xs_in_this_iteration.append(best_x)
            best_costs_in_this_iteration.append(best_cost)

        # Find the best solution of this iteration
        best_cost_index_in_this_iteration = np.argmin(best_costs_in_this_iteration)
        best_cost_in_this_iteration = best_costs_in_this_iteration[best_cost_index_in_this_iteration]
        best_x_in_this_iteration = best_xs_in_this_iteration[best_cost_index_in_this_iteration]
        if best_cost_in_this_iteration < global_best_cost:
            global_best_cost = best_cost_in_this_iteration
            global_best_position = best_x_in_this_iteration
            
        
        # Find the global best particle (for use in the velocity vector):
        if global_best_option == 'this_iteration':
            global_best_x = best_x_in_this_iteration
        elif global_best_option == 'so_far':
            global_best_x = global_best_position

        # Update every particle (using regularization hyper-parameters)
        for particle_index in range(len(particles)):
            
            phi = particles[particle_index]["phi"]
            distance_between_state_and_local_best = np.linalg.norm(particles[particle_index]['position'] - best_x_in_this_iteration) #best_x_in_this_iteration
            distance_between_state_and_global_best = np.linalg.norm(particles[particle_index]['position'] - global_best_position)
            new_particle_state = discretize_state(distance=(distance_between_state_and_local_best, distance_between_state_and_global_best), 
                                                  discrete_interval_legth=discrete_interval_legth, num_states=NUM_STATES)
            particles[particle_index]['new_state'] = new_particle_state

            if particles[particle_index]["curr_cost_for_position"] > particles[particle_index]["prev_cost"]:
                reward = -1
            elif abs(particles[particle_index]["curr_cost_for_position"] - particles[particle_index]["prev_cost"]) < 0.01:
                reward = 0
            else:
                reward = 1

            particles[particle_index]["q_table"].update_q_value(particles[particle_index]['curr_state'], 
                                                                particles[particle_index]['curr_action'], 
                                                                reward, 
                                                                particles[particle_index]["new_state"])


            action = particles[particle_index]["q_table"].select_action(particles[particle_index]["new_state"])

            if action == 0:
                phi += 0.02
            elif action == 1:
                phi -= 0.02

            if phi < 2:
                phi = 2

            alpha_1 = 1/(phi-1+np.sqrt(phi**2 - 2*phi))
            alpha_max = phi*alpha_1
            alpha_2 = alpha_max*np.random.uniform(low=0, high=1, size=1)
            alpha_3 = alpha_max*np.random.uniform(low=0, high=1, size=1)

            particles[particle_index]['velocity'] = (alpha_1 * particles[particle_index]['velocity'] +
                                                     alpha_2 * (local_best_x - particles[particle_index]['position']) +
                                                     alpha_3 * (global_best_x - particles[particle_index]['position']))
            particles[particle_index]['position'] = particles[particle_index]['position'] + particles[particle_index]['velocity']
            particles[particle_index]['position'] = bound_solution_in_x_range(x=particles[particle_index]['position'], x_range=x_range)
            particles[particle_index]['position_history'].append(particles[particle_index]['position'])

            particles[particle_index]['curr_state'] = particles[particle_index]["new_state"]
            particles[particle_index]['curr_action'] = action
            particles[particle_index]["phi"] = phi
            particles[particle_index]["prev_cost"] = particles[particle_index]["curr_cost_for_position"]

        x_history.append(global_best_position)
        cost_history.append(global_best_cost)


    for particle_index in range(len(particles)):
        np.save(file=f"particle_data/p_{particle_index}.npy", arr=particles[particle_index]['q_table'].q_table)

    return global_best_position, global_best_cost, x_history, cost_history, particles

##############################################################################################################
############ Iterative Local Search Algorithm ################################################################
##############################################################################################################
def iterative_local_search(cost_function: Callable, max_itr_ils: int, max_itr_ls: int, convergence_threshold: float,
                           x_initial: Optional[np.array] = None, x_range: Optional[List[List[float]]] = None) -> Tuple[np.array, float, List[np.array], List[float]]:
    # Set the x_initial
    if x_initial is None:
        x_initial = [random.uniform(x_range[i][0], x_range[i][1]) for i in range(len(x_range))]

    x_current = x_initial
    cost_current = cost_function(x_current)

    x_history = [x_current]
    cost_history = [cost_current]

    
    for _ in range(max_itr_ils):
        # Do local search
        best_x, best_cost, _, _ = local_search(cost_function=cost_function, max_itr=max_itr_ls, convergence_threshold=convergence_threshold,
                                               x_initial=x_current, x_range=x_range)
        x_history.append(best_x)
        cost_history.append(best_cost)
        
        # Sample from the optimization landscape
        x_current = [random.uniform(x_range[i][0], x_range[i][1]) for i in range(len(x_range))]
    
    # Get the best solution
    best_cost_index = np.argmin(cost_history)
    best_x = x_history[best_cost_index]
    best_cost = cost_history[best_cost_index]

    return best_x, best_cost, x_history, cost_history

##############################################################################################################
############ Simulated Annealing #############################################################################
##############################################################################################################
def simulated_annealing(cost_function: Callable, max_itr: int, temperature: float, alpha: float, beta: float,
                        x_initial: Optional[np.array] = None, x_range: Optional[List[List[float]]] = None,
                        temperature_decrement_method: Optional[str] = 'linear') -> Tuple[np.array, float, List[np.array], List[float]]:
    # Set the x_initial
    if x_initial is None:
        x_initial = [random.uniform(x_range[i][0], x_range[i][1]) for i in range(len(x_range))]

    x_current = x_initial
    cost_current = cost_function(x_current)

    x_history = [x_current]
    cost_history = [cost_current]

    # Set the initial temperature
    T = temperature

    # Create a tqdm progress bar

    itr = 0
    while (itr <= max_itr):
        # Generate neighboring candidates
        x_neighbor = [random.gauss(x, 0.1) for x in x_current]
        x_neighbor = bound_solution_in_x_range(x=x_neighbor, x_range=x_range)
        cost_neighbor = cost_function(x_neighbor)

        # Calculate ∆E
        Delta_E = cost_neighbor - cost_current

        # Accept the neighbor if it has lower cost
        if Delta_E <= 0:
            x_current = x_neighbor
            cost_current = cost_neighbor
            x_history.append(x_current)
            cost_history.append(cost_current)
        else:
            u = random.uniform(0, 1)
            if (u <= np.exp(-Delta_E / T)):
                x_current = x_neighbor
                cost_current = cost_neighbor
                x_history.append(x_current)
                cost_history.append(cost_current)
        
        # Decrement the temperature T
        if temperature_decrement_method == 'linear':
            T = T - alpha  # Linear reduction rule
        elif temperature_decrement_method == 'geometric':
            T = T * alpha  # Geometric reduction rule
        elif temperature_decrement_method == 'slow':
            T = T / (1 + (beta * T))  # Slow-decrease rule

        itr += 1
    
    # Get the best solution
    best_cost_index = np.argmin(cost_history)
    best_x = x_history[best_cost_index]
    best_cost = cost_history[best_cost_index]

    return best_x, best_cost, x_history, cost_history

##############################################################################################################
############ Particle Swarm Optimization (PSO) ###############################################################
##############################################################################################################
def pso(cost_function: Callable, num_particles: int, max_itr: int, alpha_1: float, alpha_2: float, alpha_3: float,
        x_initial: Optional[np.array] = None, x_range: Optional[List[List[float]]] = None,
        local_best_option: Optional[str] = 'this_iteration', global_best_option: Optional[str] = 'this_iteration',
        ls_max_itr: Optional[int] = 100, ls_convergence_threshold: Optional[float] = 0.01) -> Tuple[np.array, float, List[np.array], List[float]]:
    
    # Set the x_initial
    if x_initial is None:
        x_initial = [random.uniform(x_range[i][0], x_range[i][1]) for i in range(len(x_range))]
    
    # Initialize particles (candidate solutions)
    particles = [{'position': np.array([random.uniform(x[0], x[1]) for x in x_range]),
                  'velocity': np.array([random.uniform(-1, 1) for _ in range(len(x_range))]),
                  'best_position': x_initial,
                  'best_cost': float('inf'),
                  'position_history': []
                  } for _ in range(num_particles)]

    # Initialize global best
    global_best_position = x_initial
    global_best_cost = float('inf')
    
    x_history = []
    cost_history = []

    for _ in range(max_itr):
        best_xs_in_this_iteration, best_costs_in_this_iteration = [], []
        
        for particle_index in range(len(particles)):
            # Do local search (every particle searches locally in the local neighborhood)
            best_x, best_cost, _, _ = local_search(cost_function=cost_function, max_itr=ls_max_itr, convergence_threshold=ls_convergence_threshold,
                                                   x_initial=particles[particle_index]['position'], x_range=x_range, hide_progress_bar=True)

            # Find the local best particle (for use in the velocity vector):
            if local_best_option == 'this_iteration':
                local_best_x = best_x
            elif local_best_option == 'so_far':
                if best_cost < particles[particle_index]['best_cost']:
                    particles[particle_index]['best_cost'] = best_cost
                    particles[particle_index]['best_position'] = best_x
                    local_best_x = particles[particle_index]['best_position']
            
            best_xs_in_this_iteration.append(best_x)
            best_costs_in_this_iteration.append(best_cost)

        # Find the best solution of this iteration
        best_cost_index_in_this_iteration = np.argmin(best_costs_in_this_iteration)
        best_cost_in_this_iteration = best_costs_in_this_iteration[best_cost_index_in_this_iteration]
        best_x_in_this_iteration = best_xs_in_this_iteration[best_cost_index_in_this_iteration]
        if best_cost_in_this_iteration < global_best_cost:
            global_best_cost = best_cost_in_this_iteration
            global_best_position = best_x_in_this_iteration
        
        # Find the global best particle (for use in the velocity vector):
        if global_best_option == 'this_iteration':
            global_best_x = best_x_in_this_iteration
        elif global_best_option == 'so_far':
            global_best_x = global_best_position

        # Update every particle (using regularization hyper-parameters)
        for particle_index in range(len(particles)):
            particles[particle_index]['velocity'] = (alpha_1 * particles[particle_index]['velocity'] +
                                                     alpha_2 * (local_best_x - particles[particle_index]['position']) +
                                                     alpha_3 * (global_best_x - particles[particle_index]['position']))
            particles[particle_index]['position'] = particles[particle_index]['position'] + particles[particle_index]['velocity']
            particles[particle_index]['position'] = bound_solution_in_x_range(x=particles[particle_index]['position'], x_range=x_range)
            particles[particle_index]['position_history'].append(particles[particle_index]['position'])

        x_history.append(global_best_position)
        cost_history.append(global_best_cost)



    return global_best_position, global_best_cost, x_history, cost_history, particles

##############################################################################################################
############ Genetic Algorithm (GA) ##########################################################################
##############################################################################################################
def ga(cost_function: Callable, population_size: int, max_itr: int, mutation_rate: float, crossover_rate: float,
       x_initial: Optional[np.array] = None, x_range: Optional[List[List[float]]] = None) -> Tuple[np.array, float, List[np.array], List[float]]:
    # Initialize the population
    population = [np.array([random.uniform(r[0], r[1]) for r in x_range]) for _ in range(population_size)]


    best_solution = x_initial
    best_cost = float('inf')
    history = {'best_costs': [], 'best_solutions': []}

    # Initialize chromosome history (required for visualization)
    chromosomes = [{
                    'position_history': []
                   } for _ in range(population_size)]

    for _ in range(max_itr):
        # Evaluate the cost of each individual in the population
        cost_values = [cost_function(individual) for individual in population]  # individuals = candidate solutions

        # Update the chromosome history (required for visualization)
        for chromosome_index in range(len(chromosomes)):
            chromosomes[chromosome_index]['position_history'].append(population[chromosome_index])

        # Find the best solution in this generation/iteration
        best_generation_cost = min(cost_values)
        best_generation_index = cost_values.index(best_generation_cost)
        best_generation_solution = population[best_generation_index]
        if best_generation_cost < best_cost:
            best_solution = best_generation_solution
            best_cost = best_generation_cost
        history['best_costs'].append(best_cost)
        history['best_solutions'].append(best_solution)

        # Select parents for crossover (natural selection)
        num_parents = int(population_size * crossover_rate)  # Number of parents is selected as a fraction of population size
        parents_indices = np.argsort(cost_values)[:num_parents]  # Select the first num_parents indices corresponding to the individuals with lowest cost_values
        parents = [population[i] for i in parents_indices]

        # Create offspring through crossover
        offspring = []
        while len(offspring) < population_size:

            # Random natural selection
            parent1, parent2 = random.sample(parents, k=2)

            # One-point crossover
            crossover_point = random.randint(1, len(parent1) - 1)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            offspring.append(child)

        # Mutate offspring (random changes to the offsprings after crossover)
        for i in range(len(offspring)):
            if random.uniform(0, 1) < mutation_rate:
                mutation_point = random.randint(0, len(x_range) - 1)
                offspring[i][mutation_point] = random.uniform(x_range[mutation_point][0], x_range[mutation_point][1])

        # Replace the old population with the new population (offspring)
        population = offspring


    return best_solution, best_cost, history['best_solutions'], history['best_costs'], chromosomes

##############################################################################################################
############ Helper Functions ################################################################################
##############################################################################################################
def bound_solution_in_x_range(x: List[float], x_range: List[List[float]]) -> List[float]:
    for j in range(len(x)):
        if x[j] < x_range[j][0]:
            x[j] = x_range[j][0]
        elif x[j] > x_range[j][1]:
            x[j] = x_range[j][1]
    return x

# ZEYAD SUCKS AT COMMENTOIEDNJFSN[i pjsakljnfjilasj p[das jkjk ;aljdkcahdjflerhsjf
# 
# 
# sdfsdfsfsdkjfhsdf
# 
# 
# 
# 
# 
# This is soething that is interseting asdoinaksd asdjasdlkjadlk
# 
# 
# 
# 
# 
# ]]