import math
import sys
import numpy as np
import os
from hmmlearn.hmm import GaussianHMM

# global variables
LOWEST_FLOAT = sys.float_info.max * (-1)


def emission_calculate(x, mean, standard_deviation):
    # probability density function (The normal distribution (also called Gaussian distribution))
    return (1 / (np.sqrt(2 * math.pi) * standard_deviation)) * np.exp(
        -np.square(x - mean) / (2 * np.square(standard_deviation)))


def take_input_data(file_name):
    data = np.loadtxt(file_name, dtype=float)
    return data


def take_input_parameters(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    lines = [line.split() for line in lines]
    states = int(lines[0][0])

    trans_matrix = np.array([[float(lines[row + 1][col]) for col in range(states)] for row in range(states)])
    means_arr = np.array([float(lines[states + 1][i]) for i in range(states)])
    stds_arr = np.array([np.sqrt(float(lines[states + 2][i])) for i in range(states)])

    return trans_matrix, means_arr, stds_arr


def stationary_probability_calculation(trans_matrix):
    coefficient_mat = np.transpose(trans_matrix)
    total_equations = np.shape(trans_matrix)[0]

    for i in range(total_equations):
        coefficient_mat[i][i] -= 1

    coefficient_mat[total_equations - 1] = np.ones(total_equations)
    dependant_variables = np.zeros(total_equations)
    dependant_variables[total_equations - 1] = 1

    solutions = np.linalg.solve(coefficient_mat, dependant_variables)

    return solutions


def output_to_file(filename, row, col, matrix):
    with open(filename, "w") as file:
        for i in range(col):
            for j in range(row):
                file.write(str(matrix[j][i]) + "\t")
            file.write("\n")


# # Viterbi Implementation
def viterbi(observations, transition_matrix, means, stds, state_names_array, output_file):
    # initial probability calculation
    stationary_distribution = stationary_probability_calculation(trans_matrix=transition_matrix.copy())

    # emission calculation
    emission_matrix = np.zeros(shape=(hidden_states, total_observations), dtype=float)
    for i in range(hidden_states):
        for j in range(total_observations):
            emission_matrix[i][j] = emission_calculate(x=observations[j], mean=means[i], standard_deviation=stds[i])

    viterbi_mat = np.zeros(shape=(hidden_states, total_observations), dtype=float)
    parents = np.zeros(shape=(hidden_states, total_observations), dtype=int)

    for i in range(hidden_states):
        viterbi_mat[i][0] = np.log(stationary_distribution[i] * emission_matrix[i][0])
        parents[i][0] = -1

    for i in range(1, total_observations):
        for j in range(hidden_states):
            temp_max = LOWEST_FLOAT
            temp_parent = -1
            for k in range(hidden_states):
                temp = viterbi_mat[k][i - 1] + np.log(transition_matrix[k][j] * emission_matrix[j][i])
                if temp > temp_max:
                    temp_max = temp
                    temp_parent = k
            viterbi_mat[j][i] = temp_max
            parents[j][i] = temp_parent

    # print(viterbi_mat)
    # hidden path backtracking

    last_hidden_state = -1
    temp_last_max = LOWEST_FLOAT

    for i in range(hidden_states):
        if viterbi_mat[i][total_observations - 1] > temp_last_max:
            temp_last_max = viterbi_mat[i][total_observations - 1]
            last_hidden_state = i

    hidden_path = [last_hidden_state]
    index = total_observations - 1
    parent_index = last_hidden_state

    # print(parents[parent_index][index])
    while parents[parent_index][index] != -1:
        parent_index = parents[parent_index][index]
        index -= 1
        hidden_path.append(parent_index)

    hidden_path.reverse()

    with open(output_file, "w") as viterbi_output_file:
        for i in hidden_path:
            viterbi_output_file.write("\"" + state_names_array[i] + "\"\n")

    # with open("Output/viterbi_prob.txt", "w") as viterbi_output_file2:
    #     for i in range(total_observations):
    #         for j in range(hidden_states):
    #             viterbi_output_file2.write(str(viterbi_mat[j][i])+"\t")
    #         viterbi_output_file2.write("\n")
    # output_to_file(filename="Output/viterbi_prob.txt", row=hidden_states, col=total_observations, matrix=viterbi_mat)

    # with open("Output/emission.txt", "w") as viterbi_output_file3:
    #     for i in range(total_observations):
    #         for j in range(hidden_states):
    #             viterbi_output_file3.write(str(emission_matrix[j][i])+"\t")
    #         viterbi_output_file3.write("\n")
    # output_to_file(filename="Output/emission.txt", row=hidden_states, col=total_observations, matrix=emission_matrix)


# # Baum Welch Implementation

# ### forward calc
def forward(observations, transition_matrix, means, stds):
    # initial probability calculation
    stationary_distribution = stationary_probability_calculation(trans_matrix=transition_matrix.copy())

    # emission calculation
    emission_matrix = np.zeros(shape=(hidden_states, total_observations), dtype=float)
    for i in range(hidden_states):
        for j in range(total_observations):
            emission_matrix[i][j] = emission_calculate(x=observations[j], mean=means[i], standard_deviation=stds[i])

    forward_mat = np.zeros(shape=(hidden_states, total_observations), dtype=float)

    for i in range(hidden_states):
        forward_mat[i][0] = stationary_distribution[i] * emission_matrix[i][0]

    for i in range(1, total_observations):
        # normalize previous probabilities along column
        forward_mat[:, i - 1] /= np.sum(forward_mat[:, i - 1])
        for j in range(hidden_states):
            for k in range(hidden_states):
                forward_mat[j][i] += forward_mat[k][i - 1] * transition_matrix[k][j] * emission_matrix[j][i]

    # last column normalize
    forward_mat[:, total_observations - 1] /= np.sum(forward_mat[:, total_observations - 1])

    return forward_mat, stationary_distribution


# ### backward calc
def backward(observations, transition_matrix, means, stds):
    # emission calculation
    emission_matrix = np.zeros(shape=(hidden_states, total_observations), dtype=float)
    for i in range(hidden_states):
        for j in range(total_observations):
            emission_matrix[i][j] = emission_calculate(x=observations[j], mean=means[i], standard_deviation=stds[i])

    backward_mat = np.zeros(shape=(hidden_states, total_observations), dtype=float)

    for i in range(hidden_states):
        backward_mat[i][total_observations - 1] = 1.0

    for i in range(total_observations - 2, -1, -1):
        # normalize previous probabilities along column
        backward_mat[:, i + 1] /= np.sum(backward_mat[:, i + 1])
        for j in range(hidden_states):
            for k in range(hidden_states):
                backward_mat[j][i] += backward_mat[k][i + 1] * transition_matrix[j][k] * emission_matrix[k][i + 1]

    # first column normalize
    backward_mat[:, 0] /= np.sum(backward_mat[:, 0])
    return backward_mat


def Baum_Welch(updated_transition_matrix, updated_means_array, updated_stds_array):
    global new_transition_mat, new_means_ara, new_stds_ara
    f, stationary_dist = forward(observations=obs_ara, transition_matrix=updated_transition_matrix,
                                 means=updated_means_array,
                                 stds=updated_stds_array)
    b = backward(observations=obs_ara, transition_matrix=updated_transition_matrix, means=updated_means_array,
                 stds=updated_stds_array)

    # ## pi star and pi double star calculation

    # pi star calculation
    fsink = np.sum(f[:, total_observations - 1])
    pi_star = (f * b) / fsink

    # normalize along column
    pi_star /= np.sum(pi_star, axis=0)

    # pi double star calculation

    pi_double_star = np.zeros(shape=(hidden_states * hidden_states, total_observations - 1), dtype=float)
    # emission calculation
    emission_matrix = np.zeros(shape=(hidden_states, total_observations), dtype=float)
    for i in range(hidden_states):
        for j in range(total_observations):
            emission_matrix[i][j] = emission_calculate(x=obs_ara[j], mean=updated_means_array[i],
                                                       standard_deviation=updated_stds_array[i])

    index = -1
    for i in range(hidden_states):
        for j in range(hidden_states):
            index += 1  # row major way fill-up.....index = row
            for k in range(total_observations - 1):
                pi_double_star[index][k] = (f[i][k] * updated_transition_matrix[i][j] * emission_matrix[j][k + 1] *
                                            b[j][
                                                k + 1]) / fsink

    # normalize along column
    pi_double_star /= np.sum(pi_double_star, axis=0)

    # # M step (parameter estimation)

    # transition matrix
    new_transition_mat = np.sum(pi_double_star, axis=1).reshape(hidden_states,
                                                                hidden_states)  # axis=1 means along the row
    # normalize along row
    # new_transition_mat /= np.sum(new_transition_mat, axis=1) #variance in answer

    # noob way
    for i in range(hidden_states):
        temp_sum = np.sum(new_transition_mat[i, :])
        for j in range(hidden_states):
            new_transition_mat[i][j] /= temp_sum

    # distribution calculation
    # mean
    new_means_ara = np.matmul(pi_star, obs_ara) / np.sum(pi_star, axis=1)

    # standard deviation
    new_stds_ara = np.zeros(shape=hidden_states, dtype=float)

    for i in range(hidden_states):
        for j in range(total_observations):
            new_stds_ara[i] += pi_star[i][j] * np.square(obs_ara[j] - new_means_ara[i])

    new_stds_ara /= np.sum(pi_star, axis=1)
    new_stds_ara = np.sqrt(new_stds_ara)

    return stationary_dist


def convergence_test(prev_T, new_T, prev_mean, new_mean, prev_std, new_std):
    summation = np.sum(np.abs(new_T - prev_T)) + np.sum(np.abs(new_mean - prev_mean)) + np.sum(
        np.abs(new_std - prev_std))

    return summation < 0.00001


def learned_parameter_to_file(filename, hidden_states, new_transition_mat, new_means_ara, new_stds_ara,
                              stationary_distribution):
    with open(filename, "w") as p_out:
        p_out.write(str(hidden_states) + "\n")
        for idx in range(hidden_states):
            for j in range(hidden_states):
                p_out.write(str(new_transition_mat[idx][j]) + "\t\t")
            p_out.write("\n")
        for j in range(hidden_states):
            p_out.write(str(new_means_ara[j]) + "\t\t")
        p_out.write("\n")
        for j in range(hidden_states):
            p_out.write(str(np.square(new_stds_ara[j])) + "\t\t")
        p_out.write("\n")
        for j in range(hidden_states):
            p_out.write(str(stationary_distribution[j]) + "\t\t")
        p_out.write("\n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ## take inputs
    data_file_name = 'Input/data.txt'
    parameter_file_name = 'Input/parameters.txt'
    obs_ara = take_input_data(file_name=data_file_name)
    transition_mat, means_array, stds_array = take_input_parameters(file_name=parameter_file_name)
    state_names = ["El Nino", "La Nina"]
    total_observations = obs_ara.size
    hidden_states = np.shape(transition_mat)[0]  # like alpha, beta, gamma

    assert hidden_states == len(state_names), "hidden state count and state names count mismatch"

    # output directory created
    if not os.path.exists('Output'):
        os.makedirs('Output')

    viterbi(observations=obs_ara, transition_matrix=transition_mat, means=means_array, stds=stds_array,
            state_names_array=state_names, output_file="Output/states_Viterbi_wo_learning.txt")

    # BAUM_WELCH IMPLEMENTATION
    # convergence
    convergence_interation = -1
    max_iteration = 1000
    prev_transition_mat = []
    prev_means_ara = []
    prev_stds_ara = []

    new_transition_mat = transition_mat.copy()
    new_means_ara = means_array.copy()
    new_stds_ara = stds_array.copy()

    for i in range(max_iteration):
        prev_transition_mat = new_transition_mat.copy()
        prev_means_ara = new_means_ara.copy()
        prev_stds_ara = new_stds_ara.copy()
        stationary_distribution = Baum_Welch(updated_transition_matrix=new_transition_mat,
                                             updated_means_array=new_means_ara,
                                             updated_stds_array=new_stds_ara)

        if convergence_test(prev_T=prev_transition_mat, new_T=new_transition_mat, prev_mean=prev_means_ara,
                            new_mean=new_means_ara, prev_std=prev_stds_ara, new_std=new_stds_ara):
            convergence_interation = i
            learned_parameter_to_file("Output/parameters_learned.txt", hidden_states, new_transition_mat, new_means_ara, new_stds_ara, stationary_distribution)
            break

    print(f'Converged after iteration: {convergence_interation}')
    viterbi(observations=obs_ara, transition_matrix=new_transition_mat, means=new_means_ara, stds=new_stds_ara,
            state_names_array=state_names, output_file="Output/states_Viterbi_after_learning.txt")

    # scikit HMM learn compare
    print("\nScikit hmmlearn results:\n")

    hmm_obs = obs_ara.copy().reshape(-1, 1)
    model = GaussianHMM(n_components=hidden_states)
    model.fit(hmm_obs)

    print("Transition Matrix: ", model.transmat_)
    print("means: ", model.means_)
    print("variances: ", model.covars_)
    print("stationary: ", model.get_stationary_distribution())