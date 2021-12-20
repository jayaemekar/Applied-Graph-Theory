import matplotlib.pyplot as plt
import networkx as nx
import operator as op
import numpy as np

def network_data_preprocessing(filename_flow,filename_net):
    file_to_read = open(filename_flow)
    file_to_write = open("flow-cleaned-data.txt", "w")

    for line in file_to_read:
        line = line.replace(":","\t")
        line = line.replace(";", "\t")
        k = line.split()
        l = k[0] + "\t" + k[1] + "\t" + k[2] + "\t" + k[3] + "\t" + "\n"
        file_to_write.write(l)

    file_to_read = open(filename_net)

    file_to_write = open("net-cleaned-data.txt", "w")

    count = 0
    for line in file_to_read:
        line = line.replace(":","\t")
        line = line.replace(";", "\t")
        k = line.split()
        l = k[0] + "\t" + k[1] + "\t" + k[2] + "\t" + k[3] + "\t" + k[4] + "\t" + k[5] + "\t" + k[6] + "\t" + k[7] + "\t" + k[8] + "\t" + k[9] + "\n"
        file_to_write.write(l)
        count += 1

def static_analysis(graph):
    # Calculate Degree of network
    deg_assort = nx.degree_assortativity_coefficient(graph)
    avg_neighbour_degree = nx.average_neighbor_degree(graph)
    avg_neigh = []
    for key,val in avg_neighbour_degree.items():
        avg_neigh.append(val)

    # Triangles
    triangles = nx.triangles(graph)
    triangle_count = 0
    for key,val in triangles.items():
        if (val == 1):
            triangle_count += 1
            
    # Transitivity
    trans = nx.transitivity(graph)

    #Clustering
    clusters = nx.clustering(graph)
    cluster_count = 0
    for key,val in clusters.items():
        if (val != 0):
            cluster_count += 1
            
    # Average Clustering
    avg_clsuters = nx.average_clustering(graph)

    # Squares
    squares = nx.square_clustering(graph)
    square_count = 0
    for key,val in squares.items():
        if (val != 0):
            square_count += 1
            
    # Diameter
    dia = nx.diameter(graph)
    rad = nx.radius(graph)
            
    print('Number of triangles :', triangle_count)
    print('Transitivity :       ',trans)
    print('Number of clusters : ', cluster_count)
    print('Average clusters :   ', avg_clsuters)
    print('Number of squares :  ', square_count)
    print('Diameter:            ', dia)
    print('Radius:              ', rad)
    print('Average Neighbor degree :', sum(avg_neigh)/len(avg_neigh))
    print('Degree Assortativity    :', deg_assort)

#Plot degree distribution and the average degree distribution of a network
def degree_distribution(graph):
    deg = []
    degree_count = {}
    for i in graph.nodes():
        if (graph.degree(i) not in degree_count):
            degree_count[graph.degree(i)] = 1
        else:
            degree_count[graph.degree(i)] += 1
        deg.append(graph.degree(i))
            
    plt.hist(deg, bins=50)
    plt.ylabel('Degree Distribution')
    plt.xlabel('Degree of Node')
    plt.title('Transportation Network Degree Distribution')
    plt.savefig('Transportation_Network_Degree_Distribution', dpi = 400)
    plt.show()

def average_degree_distribution(graph):
    avg_neighbour_degree = nx.average_neighbor_degree(graph)
    avg_neigh = []
    for key,val in avg_neighbour_degree.items():
        avg_neigh.append(val)
        
    plt.hist(avg_neigh, bins=50)
    plt.ylabel('Neighbour Degree Distribution')
    plt.xlabel('Degree of Neighbor of Node')
    plt.title('Transportation Network Neighbor Degree Distribution')
    plt.savefig('Transportation_Network_Neighbor_Degree_Distribution', dpi = 400)
    plt.show()

#find centrality analysis of weighted and weighted network
def centralityAnalysis( graph ):
    #Determine the degree Centrality
    centrality = nx.degree_centrality(graph)
    sorted_centrality = sorted(((v, '{:0.6f}'.format(c)) for v, c in centrality.items()),
                            key=lambda x: x[1],reverse=True )

    #Determine the closeness centrality
    closeness_centrality = nx.closeness_centrality(graph)
    sorted_closeness = sorted(((v, '{:0.6f}'.format(c)) for v, c in closeness_centrality.items()),
                            key=lambda x: x[1],reverse=True )
    
    #Determine the eigenvector Centrality
    eigenvector= nx.eigenvector_centrality_numpy (graph)
    sorted_eigenvector = sorted(((v, '{:0.6f}'.format(c)) for v, c in eigenvector.items()),
                                key=lambda x: x[1],reverse=True)

    #Determine the betweenness Centrality
    # not allowed for weighted graph as graph type is multigraph
    betweenness = nx.betweenness_centrality(nx.read_edgelist("flow-cleaned-data.txt", data = False))
    sorted_betweenness = sorted(((v, '{:0.6f}'.format(c)) for v, c in betweenness.items()),
                                key=lambda x: x[1],reverse=True )

    #Determine the harmonic Centrality
    harmonic = nx.harmonic_centrality(graph)
    sorted_harmonic = sorted(((v, '{:0.6f}'.format(c)) for v, c in harmonic.items()),
                            key=lambda x: x[1],reverse=True)

    #Logic to print the values in the form of table

    list_of_deg_cent = []
    list_of_harm_cent = []
    list_of_eig_cent = []
    list_of_bet_cent = []
    list_of_close_cent =[]

    #Degree centrality processing
    sz = np.shape(sorted_centrality)
    for i in range(20):
        temp = sorted_centrality[i]
        ids = int(temp[0])
        list_of_deg_cent.append((ids, temp[1]))

    #Harmonic centrality processing
    sz = np.shape(sorted_harmonic)
    for i in range(20):
        temp = sorted_harmonic[i]
        ids = int(temp[0])
        list_of_harm_cent.append((ids, temp[1]))

    #Eigenvector centrality processing    
    sz = np.shape(sorted_eigenvector)
    for i in range(20):
        temp = sorted_eigenvector[i]
        ids = int(temp[0])
        list_of_eig_cent.append((ids, temp[1]))

    #Betweenness centrality processing    
    sz = np.shape(sorted_betweenness)
    for i in range(20):
        temp = sorted_betweenness[i]
        ids = int(temp[0])
        list_of_bet_cent.append((ids, temp[1]))


    #Betweenness centrality processing    
    sz = np.shape(sorted_closeness)
    for i in range(20):
        temp = sorted_closeness[i]
        ids = int(temp[0])
        list_of_close_cent.append((ids, temp[1]))

    #Plot data in the form of table
    print("Degree Centrality \t| Harmonic Centrality \t| EigenVector Centrality \t | Betweenness Centrality \t| Closeness Centrality")
    res = "\n".join("{}\t\t | {}\t\t | {}\t\t | {}\t\t | {}".format(w, x, y, z, a) for w, x, y, z ,a
                    in zip(list_of_deg_cent, list_of_harm_cent, list_of_eig_cent, list_of_bet_cent, list_of_close_cent))
    print(res)


#This function is used to print top 20 centrality nodes 
def print_top_20_centrality_nodes(unweighted_graph):
    # Top degree centrality nodes un-weighted
    nodes_degree = ['330', '303', '337', '299', '317', '266', '267',
                    '269', '273', '302', '304', '308', '341', '329',
                    '332', '333', '361', '369', '385', '373']

    nx.draw_spectral(unweighted_graph, node_size = 1)
    pos = nx.spectral_layout(unweighted_graph)
    nx.draw_networkx_nodes(unweighted_graph,pos,nodelist = nodes_degree,node_color='b',
                        node_size=200, alpha=0.9)
    plt.savefig("Transportation_Network_top_degree_unweighted", dpi = 400)
    plt.title("Transportation Network top 20 degree centrality-unweighted")
    plt.axis('off')
    plt.show()

    # Top harmonic centrality nodes un-weighted
    nodes_harmonic = ['319', '321', '317', '330', '356', '320', '318', 
                    '329', '333', '355', '316', '344', '357', '358',
                    '343', '328', '303', '315', '305', '354']

    nx.draw_spectral(unweighted_graph, node_size = 1)
    pos = nx.spectral_layout(unweighted_graph)
    nx.draw_networkx_nodes(unweighted_graph,pos,nodelist = nodes_harmonic,node_color='r',
                        node_size=200, alpha=0.9)
    plt.savefig("Transportation_Network_Anaheim_top_harmonic_unweighted", dpi = 400)
    plt.title("Transportation Network top 20 harmonic centrality-unweighted")
    plt.axis('off')
    plt.show()

    # Top betweenness centrality nodes un-weighted
    nodes_betweenness = ['358', '321', '305', '299', '333', '266', '277',
                        '315', '357', '384', '356', '319', '390', '385',
                        '386', '355', '388', '327', '317', '375']

    nx.draw_spectral(unweighted_graph, node_size = 1)
    pos = nx.spectral_layout(unweighted_graph)
    nx.draw_networkx_nodes(unweighted_graph,pos,nodelist = nodes_betweenness,node_color='g',
                        node_size=200, alpha=0.9)
    plt.savefig("Transportation_Network_Anaheim_top_betweenness_unweighted", dpi = 400)
    plt.title("Transportation Network top 20 betweenness centrality-unweighted")
    plt.axis('off')
    plt.show()

    # Top eigenvector centrality nodes un-weighted
    nodes_eigenvector = ['317', '329', '328', '330', '343', '342', '355',
                        '316', '354', '371', '372', '356', '299', '370',
                        '341', '315', '327', '387', '388', '373']

    nx.draw_spectral(unweighted_graph, node_size = 1)
    pos = nx.spectral_layout(unweighted_graph)
    nx.draw_networkx_nodes(unweighted_graph,pos,nodelist = nodes_eigenvector,node_color='y',
                        node_size=200, alpha=0.9)
    plt.savefig("Transportation_Network_Anaheim_top_eigenvector_unweighted", 
                dpi = 400)
    plt.title("Transportation Network top 20 eigenvector centrality-unweighted")
    plt.axis('off')
    plt.show()

    # Top harmonic centrality nodes weighted
    nodes_harmonic_weighted = ['20', '62', '397', '413', '14', '23', '414', 
                        '21', '15', '8', '2', '412', '22', '5', '19',
                        '380', '257', '416', '415', '254']

    nx.draw_spectral(unweighted_graph, node_size = 1)
    pos = nx.spectral_layout(unweighted_graph)
    nx.draw_networkx_nodes(unweighted_graph,pos,nodelist = nodes_harmonic_weighted,node_color='c',
                        node_size=200, alpha=0.9)
    plt.savefig("Transportation_Network_Anaheim_top_harmonic_weighted", dpi = 400)
    plt.title("Transportation Network top 20 harmonic centrality-weighted")
    plt.axis('off')
    plt.show()

def compute_betweenness_centrality_edges_unweighted(unweighted_graph):
    betweenness_edge_flow = nx.edge_current_flow_betweenness_centrality(unweighted_graph,
                                                         normalized=True,
                                                         weight=None,
                                                         solver='full')

    sorted_betweenness_edge = sorted(((v, '{:0.6f}'.format(c)) for v, c in betweenness_edge_flow.items()),
                            key=lambda x: x[1],reverse=True )
    
    #Print Top 20 edges for betweenness centrality for unweighted graph
    #Betweenness centrality processing 
    list_of_bet_cent_edge =[]

    sz = np.shape(sorted_betweenness_edge)
    for i in range(20):
        temp = sorted_betweenness_edge[i]
        ids = temp[0]
        list_of_bet_cent_edge.append((ids, temp[1]))

    #Plot data in the form of table
    print("Betweenness Centrality")
    res = "\n".join("{}".format(w) for w in zip(list_of_bet_cent_edge ))
    print(res)

    # Top 20 edges with highest betweennness centrality for unweighted graph

    edges = [('277', '266'), ('378', '361'), ('299', '277'), ('404', '405'),
            ('315', '299'), ('358', '333'), ('273', '272'), ('357', '358'),
            ('408', '407'), ('147', '148'), ('308', '295'), ('295', '294'),
            ('267', '268'), ('321', '305'), ('389', '390'), ('400', '401'),
            ('290', '269'), ('319', '303'), ('138', '139'), ('356', '357')]
    nx.draw_spectral(unweighted_graph, node_size = 1)
    pos = nx.spectral_layout(unweighted_graph)
    nx.draw_networkx_edges(unweighted_graph, pos,
                        edgelist = edges, width=5, alpha=0.9, edge_color='b')

    plt.savefig("Transportation_Network_Anaheim_betweenness_centrality_for_unweighted_graph", dpi = 400)
    plt.title("Transportation Network Anaheim betweenness centrality for unweighted graph")
    plt.axis('off')
    plt.show()


def compute_betweenness_centrality_edges_weighted(unweighted_graph,weighted_graph):
    # Compute the betweenness centrality for weighted graph
    betweenness_edge_weighted = nx.edge_current_flow_betweenness_centrality(weighted_graph,
                                                            normalized=True,
                                                            weight='cost',
                                                            solver='full')


    sorted_betweenness_edge_weighted = sorted(((v, '{:0.6f}'.format(c)) for v, c in betweenness_edge_weighted.items()),
                                key=lambda x: x[1],reverse=True )

    #Betweenness centrality processing 
    list_of_bet_cent_edge =[]

    sz = np.shape(sorted_betweenness_edge_weighted)
    for i in range(20):
        temp = sorted_betweenness_edge_weighted[i]
        ids = temp[0]
        list_of_bet_cent_edge.append((ids, temp[1]))

    #Plot data in the form of table
    print("Betweenness Centrality")
    res = "\n".join("{}".format(w) for w
                    in zip(list_of_bet_cent_edge ))
    print(res)

    # Top 20 edges with highest change in cost for weighted graph

    edges = [('400', '401'), ('147', '148'), ('230', '229'), ('136', '135'), 
            ('181', '182'), ('146', '145'), ('238', '239'), ('295', '294'), 
            ('198', '197'), ('299', '277'), ('66', '65'), ('394', '393'), 
            ('83', '84'), ('235', '236'), ('410', '409'), ('137', '138'), 
            ('72', '71'), ('58', '145'), ('147', '57'), ('115', '114')]
    nx.draw_spectral(unweighted_graph, node_size = 1)
    pos = nx.spectral_layout(unweighted_graph)
    nx.draw_networkx_edges(unweighted_graph, pos,
                        edgelist = edges, width=5, alpha=0.9, edge_color='b')

    plt.savefig("Transportation_Network_Anaheim_top_20_change_weighted_corrected", dpi = 400)
    plt.title("Transportation Network Anaheim betweenness centrality for weighted graph")
    plt.axis('off')
    plt.show()

def compute_shortest_path(graph):
    all_pairs_shortest_path = []
    for i in graph.nodes():
        for j in graph.nodes():
            if (i != j):
                all_pairs_shortest_path.append((i,j))
    return all_pairs_shortest_path

def compute_shortest_path_length(graph,all_pairs_shortest_path):
    # Compute and store all the shortest geodesic paths and path lengths
    shortest_paths_unweighted = {}
    shortest_path_lengths_unweighted = {}
    for k in all_pairs_shortest_path:
        shortest_path = nx.dijkstra_path(graph, k[0], k[1])
        shortest_path_len = nx.dijkstra_path_length(graph, k[0], k[1])  
        new = []
        for s in shortest_path:
            new.append(int(s))       
        c = int(shortest_path_len)   
        jj = (int(k[0]), int(k[1]))   
        shortest_paths_unweighted[jj] = new
        shortest_path_lengths_unweighted[jj] = c
    
    return shortest_paths_unweighted, shortest_path_lengths_unweighted


def plot_histogram_geodesic_length(all_pairs_unweighted_length):
    geo_length_values = []
    for v in all_pairs_unweighted_length.values():
        geo_length_values.append(v)

    # Histrogram for geodesic shortest path lengths with 50 bins
    plt.hist(geo_length_values, bins=50)
    plt.xlabel('Shortest Geodesic Path Lengths')
    plt.ylabel('Density')
    plt.title('Histogram for shortest path lengths in Anaheim network')
    plt.savefig('Anaheim-shortest-geodesic-path-length_hist-50-bins',dpi = 300)
    plt.show()

    # Histrogram for geodesic shortest path lengths with 25 bins
    plt.hist(geo_length_values, bins=25)
    plt.xlabel('Shortest Geodesic Path Lengths')
    plt.ylabel('Density')
    plt.title('Histogram for shortest path lengths in Anaheim network')
    plt.savefig('Anaheim-shortest-geodesic-path-length_hist-25-bins', 
                dpi = 300)
    plt.show()

def compute_shortest_path_length_cost(weighted_graph,all_pairs_weighted):
    # Compute and store all the shortest paths and 
    # shortest path lengths according to cost
    shortest_paths_weighted = {}
    shortest_path_lengths_weighted = {}
    for l in all_pairs_weighted:
        sp = nx.dijkstra_path(weighted_graph, l[0], l[1], weight = 'cost')
        spl = nx.dijkstra_path_length(weighted_graph, l[0], l[1], weight = 'cost')
        shortest_paths_weighted[l] = sp
        shortest_path_lengths_weighted[l] = spl

    return shortest_paths_weighted, shortest_path_lengths_weighted

def plot_histogram_minimum_cost_path_length(shortest_paths_weighted):
    cost_length_values = []

    for w in shortest_paths_weighted.values():
        cost_length_values.append(len(w))

    # Histograms for minimum cost path length

    # Histograms for minimum cost path length lengths with 50 bins
    plt.hist(cost_length_values, bins=50)
    plt.xlabel('Shortest Minimum Cost Path Lengths')
    plt.ylabel('Density')
    plt.title('Histogram for minimum cost lengths in transportation network')
    plt.savefig('Transportation-network-shortest-minimum-cost-path-length_hist-50-bins', 
                dpi = 300)
    plt.show()

    # Histograms for minimum cost path length

    # Histograms for minimum cost path length lengths with 25 bins
    plt.hist(cost_length_values, bins=25)
    plt.xlabel('Shortest Minimum Cost Path Lengths')
    plt.ylabel('Density')
    plt.title('Histogram for minimum cost lengths in transportation network')
    plt.savefig('Transportation-network-shortest-minimum-cost-path-length_hist-50-bins', 
                dpi = 300)
    plt.show()


def plot_equality_distribution_geidesic_min_cost_5(shortest_paths_unweighted,shortest_paths_weighted,all_pairs_unweighted):
    # Constructing the fraction table for 5 sized
    # Make 5 lists 
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    list_5 = []
    # Iterate through each vertex pair
    for (k1, v1), (k2, v2) in zip(shortest_paths_unweighted.items(), shortest_paths_weighted.items()):
        # If the geo path length == cost path length:
        if (v1 == v2):
            # If the geodesic dist between the vertices is 1-5:
            if (len(v1) >= 2 and len(v1) <= 6):
                list_1.append(k1)
            
            # If the geodesic dist between the vertices is 6-10:
            if (len(v1) >= 7 and len(v1) <= 11):
                list_2.append(k1)
                
            # If the geodesic dist between the vertices is 11-15:
            if (len(v1) >= 12 and len(v1) <= 16):
                list_3.append(k1)
            
            # If the geodesic dist between the vertices is 16-20:
            if (len(v1) >= 17 and len(v1) <= 21):
                list_4.append(k1)
                
            # If the geodesic dist between the vertices is 20-25:
            if (len(v1) >= 22 and len(v1) <= 26):
                list_5.append(k1)
                
    fraction_equal = (len(list_1)/len(all_pairs_unweighted),
                    len(list_2)/len(all_pairs_unweighted),
                    len(list_3)/len(all_pairs_unweighted),
                    len(list_4)/len(all_pairs_unweighted),
                    len(list_5)/len(all_pairs_unweighted))
    # Constructing the bar chart for 5 bins
    n_groups = 5
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.55

    opacity = 0.4

    rects1 = ax.bar(index, fraction_equal, bar_width,
                    alpha=opacity, color='b')

    ax.set_xlabel('Unweighted geodesic path lengths')
    ax.set_ylabel('Fraction (geodesic = minimum cost)')
    ax.set_title('Equality distribution for geodesic and min cost path lengths')
    ax.set_xticks(index)
    ax.set_xticklabels(('1-5', '6-10', '10-15', '15-20', '20-25'))
    ax.legend()

    fig.tight_layout()
    plt.savefig('Anaheim-Fraction-plot_geodesic=minimumcost_5-bins', 
                dpi = 300)
    plt.show()


def  plot_equality_distribution_geidesic_min_cost_25(shortest_paths_unweighted,shortest_paths_weighted,all_pairs_unweighted):
    # Constructing the fraction table for 1 sized
    # Make 25 lists 
    list1 = []
    list2 = []
    list3 = []
    list4 = []
    list5 = []
    list6 = []
    list7 = []
    list8 = []
    list9 = []
    list10 = []
    list11 = []
    list12 = []
    list13 = []
    list14 = []
    list15 = []
    list16 = []
    list17 = []
    list18 = []
    list19 = []
    list20 = []
    list21 = []
    list22 = []
    list23 = []
    list24 = []
    list25 = []

    # Iterate through each vertex pair
    for (k1, v1), (k2, v2) in zip(shortest_paths_unweighted.items(), shortest_paths_weighted.items()):
        # If the geo path length == cost path length:
        if (v1 == v2):
            # If the geodesic dist between the vertices is 1-5:
            if (len(v1) == 2):
                list1.append(k1)
            
            # If the geodesic dist between the vertices is 6-10:
            if (len(v1) == 3):
                list2.append(k1)
                
            # If the geodesic dist between the vertices is 11-15:
            if (len(v1) == 4):
                list3.append(k1)
            
            # If the geodesic dist between the vertices is 16-20:
            if (len(v1) == 5):
                list4.append(k1)
                
            # If the geodesic dist between the vertices is 20-25:
            if (len(v1) == 6):
                list5.append(k1)
            
            # If the geodesic dist between the vertices is 20-25:    
            if (len(v1) == 7):
                list6.append(k1)
            
            # If the geodesic dist between the vertices is 6-10:
            if (len(v1) == 8):
                list7.append(k1)
                
            # If the geodesic dist between the vertices is 11-15:
            if (len(v1) == 9):
                list8.append(k1)
            
            # If the geodesic dist between the vertices is 16-20:
            if (len(v1) == 10):
                list9.append(k1)
                
            # If the geodesic dist between the vertices is 20-25:
            if (len(v1) == 11):
                list10.append(k1)
                
            if (len(v1) == 12):
                list11.append(k1)
            
            # If the geodesic dist between the vertices is 6-10:
            if (len(v1) == 13):
                list12.append(k1)
                
            # If the geodesic dist between the vertices is 11-15:
            if (len(v1) == 14):
                list13.append(k1)
            
            # If the geodesic dist between the vertices is 16-20:
            if (len(v1) == 15):
                list14.append(k1)
                
            # If the geodesic dist between the vertices is 20-25:
            if (len(v1) == 16):
                list15.append(k1)
            
            # If the geodesic dist between the vertices is 20-25:
            if (len(v1) == 17):
                list16.append(k1)
            
            # If the geodesic dist between the vertices is 6-10:
            if (len(v1) == 18):
                list17.append(k1)
                
            # If the geodesic dist between the vertices is 11-15:
            if (len(v1) == 19):
                list18.append(k1)
            
            # If the geodesic dist between the vertices is 16-20:
            if (len(v1) == 20):
                list19.append(k1)
                
            # If the geodesic dist between the vertices is 20-25:
            if (len(v1) == 21):
                list20.append(k1)
                
            if (len(v1) == 22):
                list21.append(k1)
            
            # If the geodesic dist between the vertices is 6-10:
            if (len(v1) == 23):
                list22.append(k1)
                
            # If the geodesic dist between the vertices is 11-15:
            if (len(v1) == 24):
                list23.append(k1)
            
            # If the geodesic dist between the vertices is 16-20:
            if (len(v1) == 25):
                list24.append(k1)
                
            # If the geodesic dist between the vertices is 20-25:
            if (len(v1) == 26):
                list25.append(k1)
                
    fraction_equal_2 = (len(list1)/len(all_pairs_unweighted),
                        len(list2)/len(all_pairs_unweighted),
                        len(list3)/len(all_pairs_unweighted),
                        len(list4)/len(all_pairs_unweighted),
                        len(list5)/len(all_pairs_unweighted),
                        len(list6)/len(all_pairs_unweighted),
                        len(list7)/len(all_pairs_unweighted),
                        len(list8)/len(all_pairs_unweighted),
                        len(list9)/len(all_pairs_unweighted),
                        len(list10)/len(all_pairs_unweighted),
                        len(list11)/len(all_pairs_unweighted),
                        len(list12)/len(all_pairs_unweighted),
                        len(list13)/len(all_pairs_unweighted),
                        len(list14)/len(all_pairs_unweighted),
                        len(list15)/len(all_pairs_unweighted),
                        len(list16)/len(all_pairs_unweighted),
                        len(list17)/len(all_pairs_unweighted),
                        len(list18)/len(all_pairs_unweighted),
                        len(list19)/len(all_pairs_unweighted),
                        len(list20)/len(all_pairs_unweighted),
                        len(list21)/len(all_pairs_unweighted),
                        len(list22)/len(all_pairs_unweighted),
                        len(list23)/len(all_pairs_unweighted),
                        len(list24)/len(all_pairs_unweighted),
                        len(list25)/len(all_pairs_unweighted))

    # Constructing the bar chart for 25 bins
    n_groups_2 = 25
    fig, ax = plt.subplots()
    index_2 = np.arange(n_groups_2)
    bar_width_2 = 0.55

    opacity = 0.4

    rects1 = ax.bar(index_2, fraction_equal_2, bar_width_2,
                    alpha=opacity, color='r')

    ax.set_xlabel('Unweighted geodesic path lengths')
    ax.set_ylabel('Fraction (geodesic = minimum cost)')
    ax.set_title('Equality distribution for geodesic and min cost path lengths')
    ax.set_xticks(index_2)
    ax.set_xticklabels(('1', '2', '3', '4', '5' ,
                        '6', '7', '8', '9', '10',
                        '11','12','13','14','15',
                        '16','17','18','19','20',
                        '21','22','23','24','25'))
    ax.legend()

    fig.tight_layout()
    plt.savefig('Anaheim-Fraction-plot_geodesic=minimumcost_25-bins', 
                dpi = 300)
    plt.show()

#This function is used to calculate change in betweenness centrality for unweighted network
def dynamic_edge_removal_unweighted(unweighted_graph,all_pairs_unweighted):

    # Compute the betweenness centrality for unweighted graph
    bc_flow_uw = nx.edge_current_flow_betweenness_centrality(unweighted_graph,normalized=True,weight=None,solver='full')
    bc_flow_uw_correct = {}
    for (key, value) in bc_flow_uw.items():
        newkey_1 = int(key[0])
        newkey_2 = int(key[1])
        newkey = (newkey_1, newkey_2)
        bc_flow_uw_correct[newkey] = value

    sorted_bc_flow_uw = sorted(bc_flow_uw_correct.items(), key = op.itemgetter(1), reverse=True)

    # Compute and store the sum of all dijkstra shortest paths
    # divided by the number of node pairs
    cost_uw_1 = 0
    for k in all_pairs_unweighted:
        spl = nx.dijkstra_path_length(unweighted_graph, k[0], k[1])
        cost_uw_1 += spl
        
    cost_uw_bef = cost_uw_1/len(all_pairs_unweighted)

    # Compute the change in cost as a function of centraity of removed edge
    edge_number = 0
    change_in_path = {}

    x = []
    y = []
    # Go through each edge in the graph
    for edge_data in sorted_bc_flow_uw:
        edge = edge_data [0]

        # Remove the edge under inspection from graph
        unweighted_graph.remove_edge(str(edge[0]), str(edge[1]))

        # For all pairs of nodes in the graph:
        cost_uw_2 = 0
        cost_uw_aft = 0
        delta_c = 0
        for k in all_pairs_unweighted:
            # Compute and store the dijkstra shortest paths
            spl = nx.dijkstra_path_length(unweighted_graph, k[0], k[1])
            # Find sum of all the shortest paths
            cost_uw_2 += spl
        # Divide by the number of node pairs in graph to normalize
        cost_uw_aft = cost_uw_2/len(all_pairs_unweighted)
  
        # Find change delta_c from sum of all paths before edge removal
        delta_c = cost_uw_aft - cost_uw_bef
 
        # Fetch edge centrality from bc_flow_uw dictionary
        ec = bc_flow_uw_correct[edge]

        # Store centrality (key) and delta_c (value) in a dictionary
        change_in_path[(edge, ec)] = delta_c
        
        x.append(ec)
        y.append(delta_c)
        
        # Add edge (without weight) back to the graph 
        unweighted_graph.add_edge(str(edge[0]), str(edge[1]))
        
        # Count edges completed
        edge_number += 1
        print('Edge number :', edge_number)

    # Plotting the figure 
    plt.scatter(x, y, alpha=0.5, color = 'b')
    plt.xlabel('Betweenness Centrality of removed edge')
    plt.ylabel('Change in shortest path length')
    plt.title('Change in net shortest paths vs. Edge betweenness')
    plt.savefig('Change_vs_Betweenness_Anaheim_Unweighted', dpi = 300)
    plt.show()


# Function to calculate the shortest paths after removing an edge from weighted network
def dynamic_edge_removal_weighted_first_60nodes(weighted_graph,all_pairs_weighted):

    # Compute the betweenness centrality for weighted graph
    bc_flow_w = nx.edge_current_flow_betweenness_centrality(weighted_graph, normalized=True,
                                                            weight='cost',solver='full')

    sorted_bc_flow_w = sorted(bc_flow_w.items(),  key = op.itemgetter(1),  reverse=True)

    # Compute and store the sum of all dijkstra shortest paths
    # divided by the number of node pairs
    cost_w_2 = 0
    for k in all_pairs_weighted:
        spl = nx.dijkstra_path_length(weighted_graph, k[0], k[1], weight = 'cost')
        cost_w_2 += spl
        
    cost_w_bef = cost_w_2/len(all_pairs_weighted)

    # Compute the change in cost as a function of centraity of
    # removed edge
    edge_number_w = 0
    change_in_path_w = {}

    xw = []
    yw = []
    # Go through each edge in the graph
    for edge_data_w in sorted_bc_flow_w:

        edge_w = edge_data_w [0]
        #print('The edge is :', edge)
        # Remove the edge under inspection from graph
        #print(len(graph1.edges()))
        weighted_graph.remove_edge(edge_w[0], edge_w[1])
        #print(len(graph1.edges()))
        # For all pairs of nodes in the graph:
        cost_w_2 = 0
        cost_w_aft = 0
        delta_c_w = 0
        for k in all_pairs_weighted:
            # Compute and store the dijkstra shortest paths
            spl = nx.dijkstra_path_length(weighted_graph, k[0], k[1], weight = 'cost')
            # Find sum of all the shortest paths
            cost_w_2 += spl
        # Divide by the number of node pairs in graph to normalize
        cost_w_aft = cost_w_2/len(all_pairs_weighted)
        #print('Cost after' , cost_uw_aft)
        # Find change delta_c from sum of all paths before edge removal
        delta_c_w = cost_w_aft - cost_w_bef
        #print('Change :' , delta_c)
        # Fetch edge centrality from bc_flow_uw dictionary
        ec = bc_flow_w[edge_w]
        #print('Edge centrality :' , ec)
        # Store centrality (key) and delta_c (value) in a dictionary
        change_in_path_w[(edge_w, ec)] = delta_c_w
        
        xw.append(ec)
        yw.append(delta_c_w)
        
        # Add edge (without weight) back to the graph 
        weighted_graph.add_edge(edge_w[0], edge_w[1])
        #print(len(graph1.edges()))
        
        # Count edges completed
        edge_number_w += 1
        print('Edge number :', edge_number_w)
        
    # Plotting the figure 
    plt.scatter(xw, yw, alpha=0.5, color = 'b')
    plt.xlabel('Betweenness Centrality of removed edge')
    plt.ylabel('Change in shortest path length')
    plt.title('Change in net shortest paths vs. Edge betweenness Weighted-First-60-nodes')
    plt.savefig('Change_vs_Betweenness_Anaheim_Weighted-First-60-nodes', dpi = 300)
    plt.show()

# Function to calculate the shortest paths after removing an edge from weighted network
def dynamic_edge_removal_weighted(weighted_graph,all_pairs_weighted):

    # Compute the betweenness centrality for weighted graph
    bc_flow_w = nx.edge_current_flow_betweenness_centrality(weighted_graph,normalized=True,
                                                            weight='cost',solver='full')

    sorted_bc_flow_w = sorted(bc_flow_w.items(), key = op.itemgetter(1), reverse=True)
    
    # Compute and store the sum of all dijkstra shortest paths
    # divided by the number of node pairs
    shortest_paths_w = {}
    shortest_path_lengths_w = {}
    cost_w_2 = 0
    for k in all_pairs_weighted:
        sp = nx.dijkstra_path(weighted_graph, k[0], k[1], weight = 'cost')
        spl = nx.dijkstra_path_length(weighted_graph, k[0], k[1], weight = 'cost')
        cost_w_2 += spl
        shortest_paths_w[k] = sp
        shortest_path_lengths_w[k] = spl
    
    cost_w_bef = cost_w_2/len(all_pairs_weighted)

    # Compute the change in cost as a function of centraity of
    # removed edge
    edge_number_w = 0
    change_in_path_w = {}

    xw = []
    yw = []
    # Go through each edge in the graph
    for edge_data_w in sorted_bc_flow_w:
               
        edge_w = edge_data_w [0]
        #print('The edge is :', edge_w)
        
        # Remove the edge under inspection from graph
        #print(len(graph2.edges()))
        weighted_graph.remove_edge(edge_w[0], edge_w[1])
        #print(len(graph2.edges()))
        
        cost_w_2 = 0
        cost_w_aft = 0
        delta_c_w = 0
        # For all pair of nodes in the graph:
        for k in all_pairs_weighted:
            # Fetch the shortest path from the shortest_path_w dictionary
            path = shortest_paths_w[k]
            #print(path, len(path))
            # Check if edge is there in the shortest path
            m = any([edge_w[0], edge_w[1]] == path[i:i+2] for i in range(len(path) - 1))
            # If the removed edge is there in the 
            # current shortest path
            if (m == True):
                #print(m)
                # Find sum of all the shortest paths
                cost_w_2 += len(path)
                #print(len(path))
            # Else
            else: 
                #print(m)
                # Recalculate the shortest path and store the length
                spl = nx.dijkstra_path_length(weighted_graph, k[0], k[1], weight = 'cost')
                # Find sum of all the shortest paths
                cost_w_2 += spl
        # Divide by the number of node pairs in graph to normalize
        cost_w_aft = cost_w_2/len(all_pairs_weighted)
        #print('Cost after' , cost_uw_aft)
        # Find change delta_c from sum of all paths before edge removal
        delta_c_w = cost_w_aft - cost_w_bef
        #print('Change :' , delta_c_w)
        # Fetch edge centrality from bc_flow_uw dictionary
        ec = bc_flow_w[edge_w]
        #print('Edge centrality :' , ec)
        # Store centrality (key) and delta_c (value) in a dictionary
        change_in_path_w[(edge_w, ec)] = delta_c_w
        
        xw.append(ec)
        yw.append(delta_c_w)
        
        # Add edge (without weight) back to the graph 
        weighted_graph.add_edge(edge_w[0], edge_w[1])
        #print(len(graph1.edges()))
        
        # Count edges completed
        edge_number_w += 1
        print('Edge number :', edge_number_w)
        #print()
        

        
    # Plotting the figure 
    plt.scatter(xw, yw, alpha=0.5, color = 'b')
    plt.xlabel('Betweenness Centrality of removed edge')
    plt.ylabel('Change in shortest path length')
    #plt.title('Change in net shortest paths vs. Edge betweenness')
    plt.savefig('Change_vs_Betweenness_Anaheim_Weighted', dpi = 300)
    plt.show()  









if __name__=='__main__':
    network_data_preprocessing("Anaheim_flow.tntp","Anaheim_net.tntp")

    # unweighted graph of network_
    unweighted_graph = nx.read_edgelist("flow-cleaned-data.txt", data = False)
    fig = plt.figure(1, figsize=(10, 10), dpi=60)
    nx.draw_spectral(unweighted_graph)

    # This contains the edges with the volumes and the costs
    weighted_graph = nx.read_edgelist("flow-cleaned-data.txt",
                        create_using = nx.MultiGraph(),
                        nodetype = int,
                        data = [('volume',float),('cost',float)])

    fig = plt.figure(2, figsize=(10, 10), dpi=60)
    nx.draw_spectral(weighted_graph)
    #-----------------------------Static Analysis of Network-----------------

    #perform static analysis of graph
    static_analysis(unweighted_graph)

    #Plot degree distribution and the average degree distribution of a network
    degree_distribution(unweighted_graph)
    average_degree_distribution(unweighted_graph)

    #determine the centrality analysis for weighted and unweighted graph
    centralityAnalysis(unweighted_graph)
    centralityAnalysis(weighted_graph)

    #print top 20 centrality nodes on graph
    print_top_20_centrality_nodes(unweighted_graph)

    #Betweenness centrality of edges
    #Compute the betweenness centrality for unweighted graph on edge
    compute_betweenness_centrality_edges_unweighted(unweighted_graph)

    #Compute the betweenness centrality for unweighted graph on edge
    compute_betweenness_centrality_edges_weighted(unweighted_graph,weighted_graph)

    #Calculate Shortest Paths between junctions
    #Compute geodesic shortest paths for unweighted graph
    all_pairs_unweighted = compute_shortest_path(unweighted_graph)

    #Compute and store all the shortest geodesic paths and path lengths
    shortest_paths_unweighted, shortest_path_lengths_unweighted = compute_shortest_path_length(unweighted_graph,all_pairs_unweighted)

    #Histograms for geodesic shortest paths with bins 25 and 50
    plot_histogram_geodesic_length(shortest_path_lengths_unweighted)

    #Equality distribution for geodesic and min cost path lengths on weighted graph
    #Compute geodesic shortest paths for unweighted graph
    all_pairs_weighted = compute_shortest_path(weighted_graph)

    #Compute and store all the shortest paths and path lengths according to cost
    shortest_paths_weighted, shortest_path_lengths_weighted = compute_shortest_path_length_cost(weighted_graph,all_pairs_weighted)

    #Histograms for minimum cost path length
    plot_histogram_minimum_cost_path_length(shortest_paths_weighted)

    #Equality distribution for geodesic and min cost path lengths weighted vs unweighted fraction size 5
    plot_equality_distribution_geidesic_min_cost_5(shortest_paths_unweighted,shortest_paths_weighted,all_pairs_unweighted)


    #Equality distribution for geodesic and min cost path lengths weighted vs unweighted fraction size 25
    plot_equality_distribution_geidesic_min_cost_25(shortest_paths_unweighted,shortest_paths_weighted,all_pairs_unweighted)

     #-----------------------------Dynamic Analysis of Network-----------------

    # Function to calculate the shortest paths after removing an edge from unweighted network
    dynamic_edge_removal_unweighted(unweighted_graph,all_pairs_unweighted)

    # Function to calculate the shortest paths after removing an edge from weighted network
    dynamic_edge_removal_weighted_first_60nodes(weighted_graph,all_pairs_weighted)

    # Function to calculate the shortest paths after removing an edge from weighted network
    dynamic_edge_removal_weighted(weighted_graph,all_pairs_weighted)
