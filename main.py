from numpy.lib.function_base import append
import numpy as np
import random
from numpy.random import randint


population = []
number_of_population = 10
size_of_a_string = 12
k = 2
random_seed = 123
c_prob = 0.6
m_prob = 0.02
fitness_function = 0
crossoverOperator = 0
replacement_function = 0
Bisection = 0
gen = 100
w = 0

def fitness(x, fitness_function, size_of_a_string):
    fit = 0
    if fitness_function == 0:
        fit = OnemaxFitness(x)
    if fitness_function == 1:
        fit = trap_fitness(x)
    if fitness_function == 2:
        fit = interleaved_fitness(x, size_of_a_string)
    return fit

#Function for implementing onemax fitness function
def OnemaxFitness(x):
    individual_fitness = x.count("1")
    return individual_fitness

#Trap-4 fitness function is used to calculate the fitness of an individual
def trap_fitness(x):
    split_string = []
    individual_fitness = 0
    for i in range(0, len(x), 4):
        split_string.append(x[i: i + 4])
    for i in range(len(split_string)):
        if split_string[i].count('1') == 4:
            individual_fitness += 4
        elif split_string[i].count('1') == 3:
            individual_fitness += 0
        elif split_string[i].count('1') == 2:
            individual_fitness += 1
        elif split_string[i].count('1') == 0:
            individual_fitness += 3
        elif split_string[i].count('1') == 1:
            individual_fitness += 2
    return individual_fitness

#Individual trap partitions for calculating the fitness of an individual
def interleaved_fitness(x, size_of_a_string):
    individual_fitness = 0
    split_string = []
    for i in range(0, len(x), round(size_of_a_string / 4)):
        split_string.append(x[i: i + round(size_of_a_string / 4)])
    for i in range(round(size_of_a_string / 4)):
        interleaved = []
        for j in range(len(split_string)):
            interleaved.append(split_string[j][i])
            x = "".join(interleaved)
        individual_fitness += trap_fitness(x)
    return individual_fitness

#Selecting individuals using k-tournament selection
def selection(total_population, size_of_a_string):
    random1 = random.choice(total_population)
    random2 = random.choice(total_population)
    if fitness(random1, fitness_function, size_of_a_string) > fitness(random2, fitness_function, size_of_a_string):
        return random1
    return random2

def crossover(parent1, parent2, crossoverOperator, c_prob):
    if crossoverOperator == 0:
        p1, p2 = uniform_crossover(parent1, parent2, c_prob)
    elif crossoverOperator == 1:
        p1, p2 = onepoint_crossover(parent1, parent2)
    elif crossoverOperator == 2:
        p1, p2 = twopoint_crossover(parent1, parent2)
    return [p1, p2]

#Uniform crossover operator for recombination
def uniform_crossover(parent1, parent2, prob):
    parent_list1 = list(parent1)
    parent_list2 = list(parent2)
    size = min(len(parent1), len(parent2))
    for i in range(size):
        if random.random() < prob:
            parent_list1[i], parent_list2[i] = parent_list2[i], parent_list1[i]
    p1 = "".join(parent_list1)
    p2 = "".join(parent_list2)
    return [p1, p2]

#One point crossover operator for recombination
def onepoint_crossover(parent1, parent2):
    parent_list1 = list(parent1)
    parent_list2 = list(parent2)
    size = min(len(parent1), len(parent2))
    point = random.randint(0, size)
    for i in range(point, len(parent1)):
        parent_list1[i], parent_list2[i] = parent_list2[i], parent_list1[i]
    p1 = "".join(parent_list1)
    p2 = "".join(parent_list2)
    return [p1, p2]

#Teo point crossover operator for recombination
def twopoint_crossover(parent1, parent2):
    parent_list1 = list(parent1)
    parent_list2 = list(parent2)
    size = min(len(parent1), len(parent2))
    point1 = random.randint(0, size)
    point2 = random.randint(0, size)
    for i in range(size):
        if point1 <= i <= point2:
            parent_list1[i], parent_list2[i] = parent_list2[i], parent_list1[i]
    p1 = "".join(parent_list1)
    p2 = "".join(parent_list2)
    return [p1, p2]

#Bit-flip mutation
def mutation(parent1, parent2, prob):
    parent_list1 = list(parent1)
    parent_list2 = list(parent2)
    size = min(len(parent1), len(parent2))
    for i in range(size):
        if random.random() < prob:
            parent_list1[i] = "0" if parent_list1[i] == "1" else "1"
            parent_list2[i] = "0" if parent_list2[i] == "1" else "1"
        p1 = "".join(parent_list1)
        p2 = "".join(parent_list2)
    return [p1, p2]

def replacement(total_population, all_child, replacement_function, fitness_function, size_of_a_string, w):
    if replacement_function == 0:
        replace_pop = elitism_replacement(total_population, all_child)
    elif replacement_function == 1:
        replace_pop = restricted_replacement(total_population, all_child, fitness_function, size_of_a_string, w)
    else:
        print("Entered wrong replacement function")
    return replace_pop

#full replacement with elitism of one individual
def elitism_replacement(total_population, all_child):
    all_child.pop(0)
    all_child.append(total_population[-1])
    return all_child

#Restricted replacement
def restricted_replacement(total_population, all_child, fitness_function, size_of_a_string, w):
    for i in range(len(all_child)):
        x = all_child[i]
        index = i
        hamming = []
        W = random.choices(total_population, k=w)
        for l in range(len(W)):
            count = 0
            for k in range(len(x)):
                h = W[l]
                if (x[k] != h[k]):
                    count += 1
            hamming.append(count)
        for i in range(len(hamming)):
            for j in range(len(hamming) - i - 1):
                if (hamming[j] > hamming[j + 1]):
                    hamming[j], hamming[j + 1] = hamming[j + 1], hamming[j]
                    W[j], W[j + 1] = W[j + 1], W[j]
        if fitness(x, fitness_function, size_of_a_string) > fitness(W[0], fitness_function, size_of_a_string):
            for k in range(len(total_population)):
                if W[0] == total_population[k]:
                    pop_index = k
            total_population[pop_index] = all_child[index]
    return total_population

#Finding Best, Average, and Worst individuals
def best_avg_worst(all_child, size_of_a_string):
    best_fit = []
    for i in range(len(all_child)):
        fit = fitness(all_child[i], fitness_function, size_of_a_string)
        best_fit += [fit]
    for i in range(len(all_child)):
        for j in range(len(all_child) - i - 1):
            if best_fit[j] > best_fit[j + 1]:
                best_fit[j], best_fit[j + 1] = best_fit[j + 1], best_fit[j]
                all_child[j], all_child[j + 1] = all_child[j + 1], all_child[j]
    max_fitness = max(best_fit)
    min_fitness = min(best_fit)
    avg_fitness = round(((max_fitness + min_fitness) / 2))
    average = [i for i in range(len(best_fit)) if best_fit[i] == avg_fitness]
    if average in best_fit:
        average = best_fit[min(range(len(best_fit)), key=lambda i: abs(best_fit[i] - average))]
    return max_fitness, min_fitness, avg_fitness, all_child, best_fit

if w == 0:
    w = round((size_of_a_string / 10))

#Implementation of simple genetic algorithm
def Simple_GA(population_count):
    total_population = []
    best_fit = []
    optimal_fitness = []
    count = 0

    #Random initialization of population
    for pop in range(population_count):
        population = np.random.choice(['1', '0'], size_of_a_string)
        str_population = "".join(population)
        # Fitness Calculation
        x = fitness(str_population, fitness_function, size_of_a_string)
        total_population += [str_population]
    pop_best, pop_worst, pop_avg, total_population, pop_fitness = best_avg_worst(total_population, size_of_a_string)
    optimal_fitness += [[pop_best, pop_avg, pop_worst]]
    finish = True
    while finish:
        all_child = []
        child_fitness = []
        for i in range(round(population_count / 2)):
            parent = []
            for j in range(k):
                select = selection(total_population, size_of_a_string)
                parent += [select]

            # crossover with configurable probability to apply
            parent1, parent2 = crossover(parent[0], parent[1], crossoverOperator, c_prob)

            # mutation with probability of 1/n
            p1, p2 = mutation(parent1, parent2, m_prob)

            all_child.append(p1)
            all_child.append(p2)

        for i in range(len(all_child)):
            x = fitness(all_child[i], fitness_function, size_of_a_string)
            child_fitness += [x]
        child_best, child_worst, child_average, all_child, child_fitness = best_avg_worst(all_child, size_of_a_string)
        count = count + 1
        print("Generation:", count, "[B: ", child_best, ", A: ", child_average, ", W: ", child_worst, "]")

        pop_best, pop_worst, pop_avg, total_population, pop_fitness = best_avg_worst(total_population, size_of_a_string)

        #Replacement of individuals
        after_replace = replacement(total_population, all_child, replacement_function, fitness_function, size_of_a_string, w)
        total_population = after_replace

        pop_best, pop_worst, pop_avg, total_population, pop_fitness = best_avg_worst(total_population, size_of_a_string)

        #Termination Criteria
        #If global best string is found
        if pop_best == size_of_a_string:
            globalfit_individual = total_population[-1]
            finish = False

        optimal_fitness += [[child_best, child_average, child_worst]]

        #Failsafe condition - keeping track of the fitness of last three generations
        if optimal_fitness[-1][0] == optimal_fitness[-2][0] and optimal_fitness[-1][1] == optimal_fitness[-2][1] and optimal_fitness[-1][2] == optimal_fitness[-2][2]:
            if count > 1:
                if optimal_fitness[-2][0] == optimal_fitness[-3][0] and optimal_fitness[-2][1] == optimal_fitness[-3][1] and optimal_fitness[-2][2] == optimal_fitness[-3][2]:
                    print("Reached failsafe condition")
                    globalfit_individual = all_child[-1]
                    finish = False

        #Terminate of maximum number of generation reached
        if count == gen:
            print("Reached Maximum number of generations")
            globalfit_individual = all_child[-1]
            finish = False

    if fitness(globalfit_individual, fitness_function, size_of_a_string) == size_of_a_string:
        fitx = fitness(globalfit_individual, fitness_function, size_of_a_string)
    # print("Best fit string found in present generation: ", globalfit_individual)
    # print("Best fit: ", fitx)
    else:
        fity = fitness(globalfit_individual, fitness_function, size_of_a_string)
    # print("Best fit string found in present generation: ", globalfit_individual)
    # print("Best fit: ", fity)
    if fitness(globalfit_individual, fitness_function, size_of_a_string) == size_of_a_string:
        return "success"
    else:
        return "failed"

#Bisection GA - running GA using various population to automatically narrow down to the minimum population required to solve a problem
def bisection_GA(bisection, population_count):
    if bisection == 0:
        N = population_count
        Simple_GA(N)
    if bisection == 1:
        population_count = 10
        success = True
        while success:
            result = Simple_GA(population_count)
            while result == "failed":
                population_count = population_count * 2
                result = Simple_GA(population_count)
            minimum, maximum = population_count / 2, population_count
            while ((maximum - minimum) / minimum) >= 0.01:
                population_count = round((minimum + maximum) / 2)
                result = Simple_GA(population_count)
                if result == "success":
                    maximum = population_count
                else:
                    minimum = population_count
            if result == "success":
                success = False
            else:
                print("Population Not found")
                success = False
        print("Population: ", population_count)

bisection_GA(Bisection, number_of_population)
print("Executed Successfully!!")
