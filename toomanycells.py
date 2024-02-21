#########################################################
#Princess Margaret Cancer Research Tower
#Schwartz Lab
#Javier Ruiz Ramirez
#February 2024
#########################################################
#This is a Python implementation of the command line 
#tool too-many-cells.
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7439807/
#########################################################
#Questions? Email me at: javier.ruizramirez@uhn.ca
#########################################################
import networkx as nx
from scipy import sparse as sp
from scipy.io import mmread
from time import perf_counter as clock
from time import sleep
import scanpy as sc
import numpy as np
import pandas as pd
import json
import re
from sklearn.decomposition import TruncatedSVD
from typing import Optional
from anndata import AnnData
from collections import deque
import os
import subprocess

#=====================================================
class MultiIndexList(list):
    """
    This class is derived from the list class.\
            It allows the use of iterables to \
            access the list. For example: \
            L[(1,2,0)] will access item #1 \
            of the list, then item #2 of the \
            previously retrieved item, and \
            finally item #0 of that last item.\
            We use this class to store the \
            TooManyCells tree in a structure \
            composed of nested lists and dictionaries.
    """
    #=================================================
    def __getitem__(self, indices):

        if hasattr(indices, '__iter__'):
            #If the indices object is iterable
            #then traverse the list using the indices.
            obj = self
            for index in indices:
                obj = obj[index]
            return obj
        else:
            #Otherwise, just use the __getitem__ 
            #method of the parent class.
            return super().__getitem__(indices)

#=====================================================
class TooManyCells:
    """
    This class focuses on one aspect of the original\
            TooManyCells tool, the clustering.\
            Though TooManyCells also offers\
            normalization, dimensionality reduction\
            and many other features, those can be \
            applied using functions from libraries like \
            scanpy, or they can be implemented locally.\
            This implementation also allows the \
            potential of new features with respect to \
            the original TooManyCells. For example, say\
            you want to continue partitioning fibroblasts\
            until you have just one cell, even if the \
            modularity becomes negative, but for CD8+\
            T-cells you do not want to have partitions\
            with less than 100 cells. This can be\
            easily implemented with a few conditions\
            using the cell annotations in the .obs \
            data frame of the AnnData object.\

            With regards to visualization, we recommend\
            using the too-many-cells-interactive tool.\
            You can find it at:\
            https://github.com/schwartzlab-methods/\
            too-many-cells-interactive.git\
            Once installed, we provide a function that\
            requires the path to the installation folder.\
            The function is called:\
            visualize_with_tmc_interactive().


    """
    #=================================================
    def __init__(self,
            input: AnnData | str,
            output: str,
            input_is_matrix_market: bool = False,
            ):
        """
        The constructor takes the following inputs.

        :param input: Path to input directory or \
                AnnData object.
        :param output: Path to output directory.
        :param input_is_matrix_market: If true, \
                the directory should contain a \
                .mtx file, a barcodes.tsv file \
                and a genes.tsv file.

        :return: a TooManyCells object.
        :rtype: :obj:`TooManyCells`

        """

        if isinstance(input, str):
            if input.endswith('.h5ad'):
                self.A = sc.read_h5ad(input)
            else:
                self.source = input
                if input_is_matrix_market:
                    self.convert_mm_from_source_to_anndata()
                else:
                    for f in os.listdir(input):
                        if f.endswith('.h5ad'):
                            fname = os.path.join(input, f)
                            self.A = sc.read_h5ad(fname)

        elif isinstance(input, AnnData):
            self.A = input
        else:
            raise ValueError('Unexpected input type.')

        if not os.path.exists(output):
            os.makedirs(output)

        #This column of the obs data frame indicates
        #the correspondence between a cell and the 
        #leaf node of the spectral clustering tree.
        n_cols = len(self.A.obs.columns)
        self.A.obs['sp_cluster'] = -1
        self.A.obs['sp_path']    = ''

        t = self.A.obs.columns.get_loc('sp_cluster')
        self.cluster_column_index = t
        t = self.A.obs.columns.get_loc('sp_path')
        self.path_column_index = t


        self.X = self.A.X.copy()

        self.n_cells, self.n_genes = self.A.shape

        print(self.A)

        self.trunc_SVD = TruncatedSVD(
                n_components=2,
                n_iter=5,
                algorithm='randomized')

        self.Dq = deque()

        self.G = nx.DiGraph()

        self.set_of_leaf_nodes = set()

        #Map a node to the path in the
        #binary tree that connects the
        #root node to the given node.
        self.node_to_path = {}

        #Map a node to a list of indices
        #that provide access to the JSON
        #structure.
        self.node_to_j_index = {}

        #the JSON structure representation
        #of the tree.
        self.J = MultiIndexList()

        self.output = output

        self.node_counter = 0

        #The threshold for modularity to 
        #accept a given partition of a set
        #of cells.
        self.eps = 1e-9

        self.load_dot_file   = False
        self.use_twopi_cmd   = True
        self.verbose_mode    = False

    #=====================================
    def apply_tf_idf(self):
        """
        Term frequency-inverse document frequency\
                We no longer use this function \
                since the scope of this program \
                is solely doing the clustering. \
                For normalization procedures, \
                please use the tools from scanpy.
        """

        print('Applying TD-IDF transform.')

        tf_idf_normalize = TfidfTransformer(norm=None)
        self.X = tf_idf_normalize.fit_transform(self.X)

    #=====================================
    def normalize_rows(self):
        """
        Divide each row of the count matrix by its \
                Euclidean norm.
        """

        print('Normalizing rows.')


        #It's just an alias.
        mat = self.X

        for i in range(self.n_cells):
            row = mat.getrow(i)
            nz = row.data
            row_norm  = np.linalg.norm(nz)
            row = nz / row_norm
            mat.data[mat.indptr[i]:mat.indptr[i+1]] = row

    #=====================================
    def modularity_to_json(self,Q):
        return {'_item': None,
                '_significance': None,
                '_distance': Q}

    #=====================================
    def cell_to_json(self, cell_name, cell_number):
        return {'_barcode': {'unCell': cell_name},
                '_cellRow': {'unRow': cell_number}}

    #=====================================
    def cells_to_json(self,rows):
        L = []
        for row in rows:
            cell_id = self.A.obs.index[row]
            D = self.cell_to_json(cell_id, row)
            L.append(D)
        return {'_item': L,
                '_significance': None,
                '_distance': None}

    #=====================================
    def run_spectral_clustering(self):
        """
        This function computes the partitions of the \
                initial cell population and continues \
                until the modularity of the newly \
                created partitions is nonpositive.
        """

        self.t0 = clock()

        self.normalize_rows()

        node_id = self.node_counter

        #Initialize the array of cells to partition
        rows = np.array(range(self.X.shape[0]))

        #Initialize the deque
        self.Dq.append((rows, node_id))

        #Initialize the graph
        self.G.add_node(node_id, size=len(rows))

        #Path to reach root node.
        self.node_to_path[node_id] = str(node_id)

        #Indices to reach root node.
        self.node_to_j_index[node_id] = None

        #Update the node counter
        self.node_counter += 1

        while 0 < len(self.Dq):
            rows, node_id = self.Dq.popleft()
            Q,S = self.compute_partition(rows)
            current_path = self.node_to_path[node_id]
            j_index = self.node_to_j_index[node_id]
            if self.eps < Q:

                D = self.modularity_to_json(Q)
                if j_index is None:
                    self.J.append(D)
                    self.J.append([[],[]])
                    j_index = (1,)
                else:
                    self.J[j_index].append(D)
                    self.J[j_index].append([[],[]])
                    j_index += (1,)

                self.G.nodes[node_id]['Q'] = Q

                for k,indices in enumerate(S):
                    new_node = self.node_counter
                    self.G.add_node(new_node,
                            size=len(indices))
                    self.G.add_edge(node_id, new_node)
                    T = (indices, new_node)
                    self.Dq.append(T)

                    #Update path for the new node
                    new_path = current_path 
                    new_path += '/' + str(new_node) 
                    self.node_to_path[new_node] = new_path

                    seq = j_index + (k,)
                    self.node_to_j_index[new_node] = seq

                    self.node_counter += 1
            else:
                #Update the relation between a set of
                #cells and the corresponding leaf node.
                #Also include the path to reach that node.
                c = self.cluster_column_index
                self.A.obs.iloc[rows, c] = node_id

                reversed_path = current_path[::-1]
                p = self.path_column_index
                self.A.obs.iloc[rows, p] = reversed_path

                self.set_of_leaf_nodes.add(node_id)

                #Update the JSON structure for a leaf node.
                L = self.cells_to_json(rows)
                self.J[j_index].append(L)
                self.J[j_index].append([])

                #==============END OF WHILE==============

        self.tf = clock()
        delta = self.tf - self.t0
        txt = ('Elapsed time for clustering: ' +
                f'{delta:.2f} seconds.')
        print(txt)


    #=====================================
    def compute_partition(self, rows):
        """
        Compute the partition of the given set\
                of cells. The rows input \
                contains the indices of the \
                rows we are to partition. \
                The algorithm computes a truncated \
                SVD and the corresponding modularity \
                of the newly created communities.
        """

        if self.verbose_mode:
            print(f'I was given: {rows=}')

        B = self.X[rows,:]
        n_rows = len(rows) 
        ones = np.ones(n_rows)
        w = B.T.dot(ones)
        L = np.sum(w**2) - n_rows
        w = B.dot(w)
        d = 1/np.sqrt(w)
        D = sp.diags(d)
        C = D.dot(B)
        W = self.trunc_SVD.fit_transform(C)

        partition = []
        Q = 0

        mask_c1 = 0 < W[:,1]
        mask_c2 = ~mask_c1

        #If one partition has all the elements
        #then return with Q = 0.
        if mask_c1.all() or mask_c2.all():
            return (Q, partition)

        masks = [mask_c1, mask_c2]

        for mask in masks:
            n_rows_msk = mask.sum()
            partition.append(rows[mask])
            ones_msk = ones * mask
            w_msk = B.T.dot(ones_msk)
            O_c = np.sum(w_msk**2) - n_rows_msk
            L_c = ones_msk.dot(w)  - n_rows_msk
            Q += O_c / L - (L_c / L)**2

        if self.verbose_mode:
            print(f'{Q=}')
            print(f'I found: {partition=}')
            print('===========================')

        return (Q, partition)

    #=====================================
    def plot_graph(self):
        """
        Plot the branching tree.
        """

        self.t0 = clock()
        print(self.G)


        fname = 'graph.dot'
        dot_fname = os.path.join(self.output, fname)

        if self.load_dot_file:
            self.G = nx.nx_agraph.read_dot(dot_fname)
            self.G = nx.DiGraph(self.G)
            self.G = nx.convert_node_labels_to_integers(
                    self.G)
        else:
            nx.nx_agraph.write_dot(self.G, dot_fname)
            #Write cell to node data frame.
            self.write_cell_assignment_to_csv()

        self.convert_graph_to_json()

        size_list = []
        Q_list = []
        node_list = []
        for node, attr in self.G.nodes(data=True):
            node_list.append(node)
            size_list.append(attr['size'])
            if 'Q' in attr:
                Q_list.append(attr['Q'])
            else:
                Q_list.append(np.nan)

        #Write node information to CSV
        D = {'node': node_list, 'size':size_list, 'Q':Q_list}
        df = pd.DataFrame(D)
        fname = 'node_info_hm.csv'
        fname = os.path.join(self.output, fname)
        df.to_csv(fname, index=False)

        if self.use_twopi_cmd:

            fname = 'output_graph.pdf'
            fname = os.path.join(self.output, fname)

            command = ['twopi',
                    '-Groot=0',
                    '-Goverlap=true',
                    '-Granksep=2',
                    '-Tpdf',
                    dot_fname,
                    '>',
                    fname,
                    ]
            command = ' '.join(command)
            p = subprocess.call(command, shell=True)

            self.tf = clock()
            delta = self.tf - self.t0
            txt = ('Elapsed time for plotting: ' +
                    f'{delta:.2f} seconds.')
            print(txt)


    #=====================================
    def convert_mm_from_source_to_anndata(self):
        """
        This function reads the matrix.mtx file \
                located at the source directory.\
                Since we assume that the matrix \
                has the format genes x cells, we\
                transpose the matrix, then \
                convert it to the CSR format \
                and then into an AnnData object.
        """

        self.t0 = clock()

        print('Loading data from .mtx file.')
        print('Note that we assume the format:')
        print('genes=rows and cells=columns.')

        fname = None
        for f in os.listdir(self.source):
            if f.endswith('.mtx'):
                fname = f
                break

        if fname is None:
            raise ValueError('.mtx file not found.')

        fname = os.path.join(self.source, fname)
        mat = mmread(fname)
        #Remember that the input matrix has
        #genes for rows and cells for columns.
        #Thus, just transpose.
        self.A = mat.T.tocsr()

        fname = 'barcodes.tsv'
        print(f'Loading {fname}')
        fname = os.path.join(self.source, fname)
        df_barcodes = pd.read_csv(
                fname, delimiter='\t', header=None)
        barcodes = df_barcodes.loc[:,0].tolist()

        fname = 'genes.tsv'
        print(f'Loading {fname}')
        fname = os.path.join(self.source, fname)
        df_genes = pd.read_csv(
                fname, delimiter='\t', header=None)
        genes = df_genes.loc[:,0].tolist()

        self.A = AnnData(self.A)
        self.A.obs_names = barcodes
        self.A.var_names = genes

        self.tf = clock()
        delta = self.tf - self.t0
        txt = ('Elapsed time for loading: ' + 
                f'{delta:.2f} seconds.')

    #=====================================
    def write_cell_assignment_to_csv(self):
        fname = 'clusters_hm.csv'
        fname = os.path.join(self.output, fname)
        labels = ['sp_cluster','sp_path']
        df = self.A.obs[labels]
        df.index.names = ['cell']
        df = df.rename(columns={'sp_cluster':'cluster',
                                'sp_path':'path'})
        df.to_csv(fname, index=True)

    #=====================================
    def convert_graph_to_json(self):
        """
        The graph structure stored in the attribute\
                self.J has to be formatted into a \
                JSON file. This function takes care\
                of that task. The output file is \
                named 'cluster_tree_hm.json' and is\
                equivalent to the 'cluster_tree.json'\
                file produced by too-many-cells.
        """
        fname = 'cluster_tree_hm.json'
        fname = os.path.join(self.output, fname)
        s = str(self.J)
        replace_dict = {' ':'', 'None':'null', "'":'"'}
        pattern = '|'.join(replace_dict.keys())
        regexp  = re.compile(pattern)
        fun = lambda x: replace_dict[x.group(0)] 
        obj = regexp.sub(fun, s)
        with open(fname, 'w') as output_file:
            output_file.write(obj)

    #=====================================
    def generate_cell_annotation_file(self,
            column: str):
        """
        This function stores a CSV file with\
                the labels for each cell.

        :param column: Name of the\
                column in the .obs data frame of\
                the AnnData object that contains\
                the labels to be used for the tree\
                visualization. For example, cell \
                types.

        """
        fname = 'cell_annotation_labels.csv'
        #ca = cell_annotations
        ca = self.A.obs[column].copy()
        ca.index.names = ['item']
        ca = ca.rename('label')
        fname = os.path.join(self.output, fname)
        self.cell_annotations_path = fname
        ca.to_csv(fname, index=True)

    #=====================================
    def visualize_with_tmc_interactive(self,
            path_to_tmc_interactive: str,
            use_column_for_labels: str = '',
            port: int = 1137):
        """
        This function produces the visualization\
                using too-many-cells-interactive.

        :param path_to_tmc_interactive: Path to \
                the too-many-cells-interactive \
                directory.
        :param use_column_for_labels: Name of the\
                column in the .obs data frame of\
                the AnnData object that contains\
                the labels to be used in the tree\
                visualization. For example, cell \
                types.
        :param port: Port to be used to open\
                the app in your browser using\
                the address localhost:port.

        """

        fname = 'cluster_tree_hm.json'
        fname = os.path.join(self.output, fname)
        tree_path = fname
        port_str = str(port)


        bash_exec = './start-and-load.sh'

        if len(use_column_for_labels) == 0:
            label_path_str = ''
            label_path     = ''
        else:
            self.generate_cell_annotation_file(
                    use_column_for_labels)
            label_path_str = '--label-path'
            label_path     = self.cell_annotations_path

        command = [
                bash_exec,
                '--tree-path',
                tree_path,
                label_path_str,
                label_path,
                '--port',
                port_str
                ]

        command = list(filter(len,command))
        command = ' '.join(command)
        #print(command)
        final_command = (f"(cd {path_to_tmc_interactive} "
                f"&& {command})")
        #print(final_command)
        url = 'localhost:' + port_str
        txt = ("Once the app is running, just type in "
                f"your browser \n        {url}")
        print(txt)
        print("The app will start loading.")
        pause = input('Press Enter to continue ...')
        p = subprocess.call(final_command, shell=True)

    #====END=OF=CLASS=====================

#Typical usage:
#import toomanycells as tmc
#obj = tmc.TooManyCells(path_to_source, path_to_output)
#obj.run_spectral_clustering()
#obj.plot_graph()
#obj.visualize_with_tmc_interactive(
#path_to_tmc_interactive,
#column_containing_cell_annotations,
#)
