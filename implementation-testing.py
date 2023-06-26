import time
import random
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import numpy as np
import openpyxl

# A- Test generation code

def graphSets(graph):    
    if(graph.number_of_nodes() == 0):
        return []   
    if(graph.number_of_nodes() == 1):
        return [list(graph.nodes())[0]]    
    vCurrent = list(graph.nodes())[0]    
    graph2 = graph.copy()
    graph2.remove_node(vCurrent)
    res1 = graphSets(graph2)    
    for v in graph.neighbors(vCurrent):        
        if(graph2.has_node(v)):
            graph2.remove_node(v)    
    res2 = [vCurrent] + graphSets(graph2)    
    if(len(res1) > len(res2)):
        return res1
    return res2

def calculate_k_parameter(v, V0, graph):
    count = 0
    for neighbor in graph.neighbors(v):
        if neighbor in V0:
            count += 1
    return count

def calculate_m_parameter(v, graph):
    count = 0
    for neighbor1 in graph.neighbors(v):
        for neighbor2 in graph.neighbors(v):
            if neighbor1 != neighbor2 and not graph.has_edge(neighbor1, neighbor2):
                count += 1
    return count


def heuristic_max_independent_set(graph):
    V0 = set(graph.nodes())     # Set of all vertices in the graph
    S = set()                   # Independent set
    Est = 0                     # Estimate for deviation from the exact solution
    maxM = 0                    # Maximum m value encountred

    while V0:   # Iterate while there are vertices remaining in V0
        k = {}  # Dictionary to store the values of parameter k for vertices
        m = {}  # Dictionary to store the values of parameter m for vertices

        for v in V0:
            k[v] = calculate_k_parameter(v, V0, graph)  # k[v] is the number of neighbors of v that are still in V0
            m[v] = calculate_m_parameter(v, graph)      # m[v] is the number of edges that need to be added to v's neighborhood to make it a complete induced subgraph, it is missing edges amoung all possible edges.

        v0 = min(V0, key=lambda v: (m[v], -k[v]))   # MinMax(m,k) Among vertices with the minimum m the vertex with the maximum k is selected
        S.add(v0)                                   # Add v0 to the independent set S
        Est += m[v0]                                # Update the estimation by adding m[v0]
        if(m[v0] > maxM):
            maxM = m[v0]

        V0.remove(v0)                   # Remove v0 from V0
        V0 -= set(graph.neighbors(v0))  # Remove the neighborhood of v0 from V0

    return S, Est, maxM

def generate_random_graph(num_vertices, edge_probability):
    graph = nx.Graph()
    graph.add_nodes_from(range(num_vertices))
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            if random.random() < edge_probability:
                graph.add_edge(i, j)
    return graph

def visualize_graph(graph, independent_set=None, ax=None, title=None, r_set = None, est=None, maxM=None):
    pos = nx.kamada_kawai_layout(graph)
    nx.draw(graph, pos=pos, with_labels=True, ax=ax)
    if independent_set:
        nx.draw_networkx_nodes(graph, pos=pos, nodelist=independent_set, node_color='r', ax=ax)
    ax.set_frame_on(True)
    ax.patch.set_edgecolor('black')
    ax.patch.set_linewidth('2')
    # Print graph description to console
    #print(f"Graph {title} Edges:")
    #print(*graph.edges, sep=',')

    if independent_set:
        is_cardinality = len(independent_set)
        is_cardinality_real = len(r_set)
        is_string = ', '.join(str(node) for node in independent_set)
        is_string_real = ', '.join(str(node) for node in r_set)
        ax.text(0.5, -0.1, f"Found Set: {is_string}\nCardinality: {is_cardinality} Expected: {is_cardinality_real} Est:{est} Max-M:{maxM}\nEdges: {graph.edges}\n", transform=ax.transAxes, ha='center', va='top')


# generate random sample input
num_rows = 2
num_cols = 2
num_vertices = 15
edge_probability = 0.25

graphs = [generate_random_graph(num_vertices, edge_probability) for _ in range(num_rows * num_cols)]


# Comment out to run algorithm and visualize solutions
#fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))

#for i in range(num_rows):
#    for j in range(num_cols):
#        idx = i * num_cols + j
#        graph = graphs[idx]
#        independent_set_real = graphSets(graph)
#        independent_set_heur, est, maxM = heuristic_max_independent_set(graph)
#        title = f"Sample {idx+1}"
#        visualize_graph(graph, independent_set_heur, ax=axs[i][j], title=title, r_set=independent_set_real,est=est, maxM=maxM)

#plt.tight_layout()
#plt.show()


#B- Performance Test Code

def run_test(graph):
    start_time = time.time()
    heuristic_max_independent_set(graph)
    end_time = time.time()
    return end_time - start_time

# Define the vertex numbers for the tests
vertex_numbers = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

# Perform 10 tests for each vertex number
num_tests = 30

# Confidence level
confidence_level = 0.95

# Create a dictionary to store the results
results = {"Vertex Number": [], "Mean Running Time (s)": [], "Standard Deviation": [], "Estimated Standard Error": [],
           "95% CI Lower Bound": [], "95% CI Upper Bound": []}

for vertex_number in vertex_numbers:
    running_times = []

    for _ in range(num_tests):
        # Generate the graph with the specified vertex number for each test
        # You need to implement a function to generate the graph based on the vertex number
        graph = generate_random_graph(vertex_number,0.5)

        # Run the test and record the running time
        test_time = run_test(graph)
        running_times.append(test_time)

    # Calculate mean, standard deviation, and estimated standard error
    mean_time = sum(running_times) / num_tests
    std_dev = stats.tstd(running_times)
    std_err = std_dev / (num_tests ** 0.5)

    # Calculate confidence interval
    t_value = stats.t.ppf(1 - (1 - confidence_level) / 2, num_tests - 1)
    ci_lower = mean_time - t_value * std_err
    ci_upper = mean_time + t_value * std_err

    # Add the results to the dictionary
    results["Vertex Number"].append(vertex_number)
    results["Mean Running Time (s)"].append(mean_time)
    results["Standard Deviation"].append(std_dev)
    results["Estimated Standard Error"].append(std_err)
    results["95% CI Lower Bound"].append(ci_lower)
    results["95% CI Upper Bound"].append(ci_upper)

#Comment out do performance tests

# Create a DataFrame from the results
#df = pd.DataFrame(results)

# Export the DataFrame as an Excel file
#df.to_excel("test_results.xlsx", index=False)

# Plot normal graph
#plt.figure()
#plt.plot(df['Vertex Number'], df['Mean Running Time (s)'], 'bo')
#plt.xlabel('Vertex Number')
#plt.ylabel('Mean Running Time (s)')
#plt.title('Mean Running Time vs. Vertex Number')


# Log line fit anylsis
#x = df['Vertex Number']
#y = df['Mean Running Time (s)']

# Take the logarithm of y-values
#log_y = np.log2(y)

# Perform linear regression
#coefficients = np.polyfit(x, log_y, 1)
#slope = coefficients[0]
#intercept = coefficients[1]

# Generate line data for plotting
#line_x = np.linspace(x.min(), x.max(), 100)
#line_y = 2**(slope * line_x + intercept)

# Plot the data and fitted line
#plt.figure()
#plt.semilogy(x, y, "o", label='Data')
#plt.semilogy(line_x, line_y, 'r-', label='Fitted Line')
#plt.xlabel('Vertex Number')
#plt.ylabel('Log Mean Running Time (s)')
#plt.title('Log Mean Running Time vs. Vertex Number')

#plt.legend()
#plt.show()

#END PERFORMANCE TESTS

#Correctness test code

# Function to calculate the cardinality of brute force algorithm
def bruteForceCardinality(graph):
    return len(graphSets(graph))

# Function to calculate the ratio of approximation algorithm
def calculateRatio(graph):
    approximation_set, _, _ = heuristic_max_independent_set(graph)
    return  bruteForceCardinality(graph) / len(approximation_set)

# Function to calculate the maximum and average cardinality difference
def calculateCardinalityDifference(graph):
    brute_force_cardinality = bruteForceCardinality(graph)
    approximation_set, _, MaxM = heuristic_max_independent_set(graph) 
    approximation_cardinality = len(approximation_set)
    max_difference = brute_force_cardinality - approximation_cardinality
    return max_difference, MaxM

# Function to perform the tests and save the results to Excel
def performTestsCorrectnes():
    # Create an Excel workbook and set up the worksheet
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.title = "Test Results"
    worksheet.append(["Vertex Number", "Ratio", "Max Cardinality Difference", "Ratio Bound"])

    # Perform the tests for different vertex sizes
    vertex_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    num_tests = 50

    ratio_values = []
    ratio_bound_values = []

    for vertex_size in vertex_sizes:
        ratio_sum = 0
        ratio_bound_sum = 0
        max_difference_max = float('-inf')

        for _ in range(num_tests):
            graph = generate_random_graph(vertex_size, 0.5)
            ratio = calculateRatio(graph)
            max_difference, MaxM = calculateCardinalityDifference(graph)
            ratio_bound = 1 + MaxM / bruteForceCardinality(graph)
            assert(ratio_bound >= ratio)

            ratio_sum += ratio
            ratio_bound_sum += ratio_bound
            max_difference_max = max(max_difference_max, max_difference)
            

        average_ratio = ratio_sum / num_tests
        average_ratio_bound = ratio_bound_sum / num_tests

        worksheet.append([vertex_size, average_ratio, max_difference_max, average_ratio_bound])
        ratio_values.append(average_ratio)
        ratio_bound_values.append(average_ratio_bound)

    # Save the workbook to a file
    workbook.save("test_results_2.xlsx")

    # Draw the plot of average ratio and ratio bound values
    plt.plot(vertex_sizes, ratio_values, label="Average Ratio")
    plt.plot(vertex_sizes, ratio_bound_values, label="Average Ratio Bound")
    plt.xlabel("Vertex Number")
    plt.ylabel("Value")
    plt.title("Average Ratio and Ratio Bound")
    plt.legend()
    plt.show()

# Comment out to Perform the tests correctness
#performTestsCorrectnes()
