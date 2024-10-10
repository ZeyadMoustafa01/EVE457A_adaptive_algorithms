import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from utils import search_algorithms, cost_functions, plot_utils

cost_function_mapping = {"sphere": "F1", 
                         "schwefel": "F2", 
                         "schaffer": "F3", 
                         "func7": "F4", 
                         "func8": "F5", 
                         "func9": "F6", 
                         "bent_cigar_function": "F7", 
                         "rastrigin_function": "F8", 
                         "high_conditioned_elliptic_function": "F9", 
                         "hgbat_function": "F10", 
                         "rosenbrocks_function": "F11", 
                         "griewank": "F12", 
                         "ackleys_function": "F13"}

def run_optimization(config: Dict):
    # Get the cost function
    if config['cost_function'] == 'sphere':
        cost_function = cost_functions.sphere
        x_range = [[-100, 100] for i in range(config['dimension'])]  # The range for each dimension
    
    elif config['cost_function'] == 'bent_cigar_function':
        cost_function = cost_functions.bent_cigar_function
        x_range = [[-100, 100] for i in range(config['dimension'])]  # The range for each dimension
    
    elif config['cost_function'] == 'rastrigin_function':
        cost_function = cost_functions.rastrigin_function
        x_range = [[-100, 100] for i in range(config['dimension'])]  # The range for each dimension
    
    elif config['cost_function'] == 'high_conditioned_elliptic_function':
        cost_function = cost_functions.high_conditioned_elliptic_function
        x_range = [[-100, 100] for i in range(config['dimension'])]  # The range for each dimension
    
    elif config['cost_function'] == 'hgbat_function':
        cost_function = cost_functions.hgbat_function
        x_range = [[-100, 100] for i in range(config['dimension'])]  # The range for each dimension
    
    elif config['cost_function'] == 'rosenbrocks_function':
        cost_function = cost_functions.rosenbrocks_function
        x_range = [[-100, 100] for i in range(config['dimension'])]  # The range for each dimension
    
    elif config['cost_function'] == 'griewank':
        cost_function = cost_functions.griewank
        x_range = [[-100, 100] for i in range(config['dimension'])]  # The range for each dimension
    
    elif config['cost_function'] == 'ackleys_function':
        cost_function = cost_functions.ackleys_function
        x_range = [[-100, 100] for i in range(config['dimension'])]  # The range for each dimension
    
    elif config['cost_function'] == 'schwefel':
        cost_function = cost_functions.schwefel
        x_range = [[-500, 500] for i in range(config['dimension'])]  # The range for each dimension
    
    elif config['cost_function'] == 'schaffer':
        cost_function = cost_functions.schaffer
        x_range = [[-100, 100] for i in range(config['dimension'])]  # The range for each dimension
    
    elif config['cost_function'] == 'griewank':
        cost_function = cost_functions.griewank
        x_range = [[-100, 100] for i in range(config['dimension'])]  # The range for each dimension
    
    elif config['cost_function'] == 'func7':
        cost_function = cost_functions.func7
        x_range = [[-1000, 1000] for i in range(config['dimension'])]  # The range for each dimension
    
    elif config['cost_function'] == 'func8':
        cost_function = cost_functions.func8
        x_range = [[-32, 32] for i in range(config['dimension'])]  # The range for each dimension
    
    elif config['cost_function'] == 'func9':
        cost_function = cost_functions.func9
        x_range = [[-5, 5] for i in range(config['dimension'])]  # The range for each dimension
    
    elif config['cost_function'] == 'func11':
        cost_function = cost_functions.func11
        x_range = [[-0.5, 0.5] for i in range(config['dimension'])]  # The range for each dimension
    
    elif config['cost_function'] == 'revenue':
        cost_function = cost_functions.revenue
        x_range = [[0, 20000] for i in range(config['dimension'])]  # The range for each dimension (it must be 2 for this function)

    # Get the search algorithm
    if config['search_algorithm'] == 'local_search':
        best_x, best_cost, x_history, cost_history = search_algorithms.local_search(cost_function=cost_function, max_itr=config['local_search']['max_itr'],
                                                                                    convergence_threshold=config['local_search']['convergence_threshold'],
                                                                                    x_initial=config['x_initial'], x_range=x_range)
    elif config['search_algorithm'] == 'iterative_local_search':
        best_x, best_cost, x_history, cost_history = search_algorithms.iterative_local_search(cost_function=cost_function, max_itr_ils=config['iterative_local_search']['max_itr_ils'],
                                                                                              max_itr_ls=config['iterative_local_search']['max_itr_ls'],
                                                                                              convergence_threshold=config['iterative_local_search']['convergence_threshold'],
                                                                                              x_initial=config['x_initial'], x_range=x_range)
    elif config['search_algorithm'] == 'simulated_annealing':
        best_x, best_cost, x_history, cost_history = search_algorithms.simulated_annealing(cost_function=cost_function, max_itr=config['simulated_annealing']['max_itr'],
                                                                                           temperature=config['simulated_annealing']['temperature'],
                                                                                           alpha=config['simulated_annealing']['alpha'],
                                                                                           beta=config['simulated_annealing']['beta'],
                                                                                           x_initial=config['x_initial'], x_range=x_range,
                                                                                           temperature_decrement_method=config['simulated_annealing']['temperature_decrement_method'])
    elif config['search_algorithm'] == 'pso':
        best_x, best_cost, x_history, cost_history, individuals = search_algorithms.pso(cost_function=cost_function, num_particles=config['pso']['num_particles'], max_itr=config['pso']['max_itr'],
                                                                                        alpha_1=config['pso']['alpha_1'], alpha_2=config['pso']['alpha_2'], alpha_3=config['pso']['alpha_3'],
                                                                                        x_initial=config['x_initial'], x_range=x_range,
                                                                                        local_best_option=config['pso']['local_best_option'],
                                                                                        global_best_option=config['pso']['global_best_option'],
                                                                                        ls_max_itr=config['pso']['local_search']['max_itr'], ls_convergence_threshold=config['pso']['local_search']['convergence_threshold'])
    
    elif config['search_algorithm'] == 'qlearning_pso':
        best_x, best_cost, x_history, cost_history, individuals = search_algorithms.qlearning_pso(cost_function=cost_function, num_particles=config['qlearning_pso']['num_particles'], max_itr=config['qlearning_pso']['max_itr'],
                                                                                        alpha_1=config['qlearning_pso']['alpha_1'], alpha_2=config['qlearning_pso']['alpha_2'], alpha_3=config['qlearning_pso']['alpha_3'],
                                                                                        x_initial=config['x_initial'], x_range=x_range,
                                                                                        local_best_option=config['qlearning_pso']['local_best_option'],
                                                                                        global_best_option=config['qlearning_pso']['global_best_option'],
                                                                                        ls_max_itr=config['qlearning_pso']['local_search']['max_itr'], ls_convergence_threshold=config['qlearning_pso']['local_search']['convergence_threshold'])
    
    elif config['search_algorithm'] == 'ga':
        best_x, best_cost, x_history, cost_history, individuals = search_algorithms.ga(cost_function=cost_function, population_size=config['ga']['population_size'], max_itr=config['ga']['max_itr'],
                                                                                       mutation_rate=config['ga']['mutation_rate'], crossover_rate=config['ga']['crossover_rate'], x_initial=config['x_initial'],
                                                                                       x_range=x_range)

    if len(best_x) == 2: 
        # If the dimensionality is 2, visualize the results.
        plot_utils.plot_results(best_x=best_x, best_cost=best_cost,
                                x_history=x_history, cost_history=cost_history,
                                cost_function=cost_function, x_range=x_range)
        if (config['search_algorithm'] == 'pso') or (config['search_algorithm'] == 'ga'):
            plot_utils.plot_results_with_population(best_x=best_x, individuals=individuals,
                                                    cost_function=cost_function, x_range=x_range)
        
        if config['cost_function'] == 'revenue':
            print(f"Cost of trucks: {best_x[0]}")
            print(f"Cost of sedans: {best_x[1]}")
    
    return best_x, best_cost, x_history, cost_history

def compare_algorithms() -> None:
    with open('./config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    algorithms = ['qlearning_pso', 'pso', 'ga', 'iterative_local_search', 'simulated_annealing']
    # cost_functions = ["sphere", "schwefel", "schaffer", "func7", "func8", "func9"]
    cost_functions = ["bent_cigar_function", "rastrigin_function", "high_conditioned_elliptic_function", "hgbat_function", "rosenbrocks_function", "griewank", "ackleys_function"]
    NUM_ITER = 50
    
    for func in cost_functions:
        print(f"cost_function = {func}")
        data = {"qlearning_pso": [], "pso": [], "ga": [], "iterative_local_search": [], "simulated_annealing": []}
        config["cost_function"] = func
        for iter in range(NUM_ITER):
            for algo in algorithms:
                config['search_algorithm'] = algo
                best_x, best_cost, x_history, cost_history = run_optimization(config=config)
                data[algo].append(best_cost)

        results = {algo: {"mean": np.mean(a=data[algo]), "std": np.std(a=data[algo]), "best": np.min(a=data[algo])} for algo in algorithms}
        df: pd.DataFrame = pd.DataFrame(data=results).transpose()
        df.to_csv(path_or_buf=f"results/benchmark_functions/{config['cost_function']}.csv")
    
    return

def create_convergence_graphs() -> None:
    with open('./config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    algorithms = ['qlearning_pso', 'pso', 'ga', 'iterative_local_search', 'simulated_annealing']

    # cost_functions = ["sphere", "schwefel", "schaffer", "func7", "func8", "func9"]
    cost_functions = ["bent_cigar_function", "rastrigin_function", "high_conditioned_elliptic_function", 
                      "hgbat_function", "rosenbrocks_function", "griewank", "ackleys_function"]
    
    for func in cost_functions:
        print(f"cost_function = {func}")
        data = {"qlearning_pso": [], "pso": [], "ga": [], "iterative_local_search": [], "simulated_annealing": []}
        config["cost_function"] = func
        for algo in algorithms:
            config['search_algorithm'] = algo
            best_x, best_cost, x_history, cost_history = run_optimization(config=config)
            data[algo] = cost_history
        
        
        plt.figure(figsize=(8, 6))
        for algo in algorithms:
            iterations = [i for i in range(len(data[algo]))]
            plt.plot(iterations, data[algo], label=algo)
        plt.title(label=f"Cost history of a single run of Metaheuristic algorithms for {cost_function_mapping[func]}")
        plt.ylabel(ylabel="Cost")
        plt.xlabel(xlabel="Iterations")
        plt.legend()
        plt.grid()
        plt.savefig(f"graphs/benchmark_functions/{cost_function_mapping[func]}")
    
    return

if __name__ == '__main__':
    # compare_algorithms()
    create_convergence_graphs()
    