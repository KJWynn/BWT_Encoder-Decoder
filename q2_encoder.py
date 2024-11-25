"""
Name: Khor Jia Wynn
Student ID: 33033919
"""
import sys
import math
from bitarray  import bitarray
class MinHeap:
    """
    Modified from provided FIT1008 implementation by Brendon Taylor and Jackson Goerner

    Min Heap implementation using a Python in-built list.
    Parent node's index is (child node's index - 1) // 2.
    Left child's index = 2 * node_idx + 1, right child index = 2 * node_idx + 2
    Cost of comparison between integers is assumed O(1) and not shown in function complexity

    Attributes:
        self.array: The array to store items using a Python list

    """

    def __init__(self, size):
        """
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        self.array = [None for i in range(size)]
        self.num_elems = 0

    def print_heap(self):
        for i in range(self.num_elems):
            print(self.array[i])

    def rise(self, added_item_idx):
        """
        Rise the added item to its correct position

        :Input: 
            added_item_idx: The item's index in the heap array that might need to rise
        :Post condition: Heap ordering is observed
        :Time complexity: O(log N), where N is the number of nodes in the heap
        :Aux space complexity: O(1)
        """

        added_item = self.array[added_item_idx]
        while added_item_idx > 0 and self.array[added_item_idx].item <self.array[(added_item_idx - 1) // 2].item:
            # swap added element with its parent if added element is smaller than its parent node
            self.array[added_item_idx], self.array[(added_item_idx - 1) // 2] = self.array[(added_item_idx - 1) // 2], self.array[added_item_idx]
            # also update index(location in heap array)
            self.array[added_item_idx].index, self.array[(added_item_idx - 1) // 2].index = self.array[(added_item_idx - 1) // 2].index, self.array[added_item_idx].index
            # update swapped element's index
            added_item_idx = (added_item_idx - 1) // 2

        self.array[added_item_idx] = added_item

    def sink(self, outofplace_item_idx):
        """
        Sink the out-of-place item to its correct position. Also updates index of item(position in heap)

        :Input:
            outofplace_item_idx: Index of item that was swapped to the root during the add operation
        :Post condition: Heap ordering is observed
        :Time complexity: O(log N), where N is the number of nodes in the heap
        :Aux space complexity: O(1)
        """
        # save the item
        outofplace_item = self.array[outofplace_item_idx]
        saved_smallestchild_index = int()

        # get the final position for the outofplace item
        while 2 * outofplace_item_idx + 2 <= self.num_elems:  # while there is child nodes to check
            smallest_idx = self.smallest_child(outofplace_item_idx)
            if self.array[smallest_idx].item >= outofplace_item.item:
                break

            saved_oop_index = outofplace_item_idx # 0 4
            saved_smallestchild_index = smallest_idx # 1

            self.array[outofplace_item_idx] = self.array[smallest_idx]  # rise the smallest child
            
            # update index of the child that rises
            self.array[outofplace_item_idx].index = saved_oop_index

            outofplace_item_idx = smallest_idx

         # place outofplace item into hole
        self.array[outofplace_item_idx] = outofplace_item
        # update index of outofplace item
        self.array[outofplace_item_idx].index = saved_smallestchild_index

    def smallest_child(self, item_idx):
        """
        Returns the index of item's child that has the smaller value.

        :Input:
            item_idx: Index of item whose smallest child we want to find
        :Precondition: 1 <= item_idx <= len(self.array) // 2 - 1
        :Return: Index of smallest child
        :Time complexity: O(1)
        :Aux space complexity: O(1)
        """
        left_idx = 2 * item_idx + 1
        right_idx = 2 * item_idx + 2
        if left_idx == len(self.array) - 1 or \
                self.array[left_idx].item < self.array[right_idx].item:
            return left_idx
        else:
            return right_idx

    def add(self, item):
        """
        Add an item into the heap, then rises the item to its correct postion
        
        :Input:
            item: Item to be added to heap
        :Post condition: Heap ordering is observed
        :Time complexity: O(log N), where N is the number of nodes in the heap

        """
        self.array[item.index] = item # O(1)
        self.rise(item.index) # O(log N)
        self.num_elems += 1



    def getMin(self):
        """
        Replaces minimum item(root) with last item, then sinks last item into its correct position, returns minimum item.
        
        :Return: The minimum item in the heap(root)
        :Time complexity: O(log N), where N is the number of nodes in the heap
        :Aux space complexity: O(1)
        """
        
        if len(self.array) == 0:
            elem_to_return = None
            return elem_to_return
        elif len(self.array) == 1:
            elem_to_return = self.array[self.num_elems-1]
            self.num_elems -= 1
            return elem_to_return
        else:
            # save minimum item
            min_item = self.array[0]

            # put last item into minimum item's place
            last_item = self.array[self.num_elems-1]
            last_item.index = 0
            self.array[0] = last_item
            self.num_elems -= 1

            # last item may be out of place, so sink it to correct position, O(log N)
            self.sink(0)


            return min_item

def bwt(string, suffix_arr):
    bwt_string = [None] * len(suffix_arr)
    for i in range(len(suffix_arr)):
        bwt_string[i] = string[(suffix_arr[i]-1)-1]
    return bwt_string


def huffmanEncoding(string):
    # compute char frequencies
    char_freqs = [None] * (126-36+1) # contains tuples of the vertex and the bitarray encoding
    nunique = 0 # number of unique characters
    encodings = [None] * (126-36+1)
    
    init_vertexes = []
    for i in range(len(string)):
        if char_freqs[ord(string[i])-36] is None:
            newVertex = Vertex(1, string[i])
            char_freqs[ord(string[i])-36] = newVertex
            nunique += 1
            init_vertexes.append(newVertex)
        else:
            char_freqs[ord(string[i])-36].item += 1
    minHeap = MinHeap(nunique)


    for i in range(len(init_vertexes)):
        init_vertexes[i].index = i
        minHeap.add(init_vertexes[i])
    min1=minHeap.getMin()
    min2=minHeap.getMin()

    huffmanTreeVertex = Vertex(min1.item+min2.item, min1.string+min2.string)
    huffmanTreeVertex.index = minHeap.num_elems
    if min1.item <= min2.item:
        huffmanTreeVertex.left = min1
        min1.bit = 0
        huffmanTreeVertex.right = min2
        min2.bit = 1
    else:
        huffmanTreeVertex.left = min2
        min2.bit = 0
        huffmanTreeVertex.right = min1
        min1.bit = 1
    minHeap.add(huffmanTreeVertex)
    nodes = [huffmanTreeVertex]
    while len(huffmanTreeVertex.string) != nunique:
        min1=minHeap.getMin()
        min2=minHeap.getMin()
        huffmanTreeVertex = Vertex(min1.item+min2.item, min1.string+min2.string)
        huffmanTreeVertex.index = minHeap.num_elems
        if min1.item <= min2.item:
            huffmanTreeVertex.left = min1
            min1.bit = 0
            huffmanTreeVertex.right = min2
            min2.bit = 1
        else:
            huffmanTreeVertex.left = min2
            min2.bit = 0
            huffmanTreeVertex.right = min1
            min1.bit = 1
        nodes.append(huffmanTreeVertex)
        minHeap.add(huffmanTreeVertex)

    leaveNodes = []


    # depth first search 
    dfs(huffmanTreeVertex, leaveNodes, nunique)
    # store encodings 
    for node in leaveNodes:
        node.bitArr = bitarray(node.depth)
        i = node.depth-1
        currNode  = node
        while i>=0:
            node.bitArr[i] = currNode.bit
            currNode = currNode.previous
            i -= 1
        
        encodings[ord(node.string)-36] = node.bitArr

    return encodings

def dfs(node, leaves, nunique, depth=0):

    # Recursively traverse each child
    if node.left is not None:
        node.left.previous = node
        dfs(node.left,leaves, nunique, depth+1)
    if node.right is not None:
        node.right.previous = node
        dfs(node.right,leaves, nunique, depth+1)

    # leaf nodes need to be processed for their bitarray
    if node.left is None and node.right is None:
        node.depth = depth
        leaves.append(node)


def minimal_bin_code(numbits_required, integer):
    bitArr = bitarray(numbits_required)
    index = numbits_required-1
    while integer > 0:
        bitArr[index] = integer%2
        integer //=2
        index -= 1
    return bitArr
    

def elias_omega(n):
    """
    Returns tuple of length components and code component for elias omega code of a positive integer
    Note that the length components are in descending order
    """
    # bitArray 
    bitArr_size = math.ceil(math.log2(n+1))
    # compute minimal binary code of the integer
    code_comp = minimal_bin_code(bitArr_size, n)
    
    # compute length components
    length_components = []
    curr_length_component = bitArr_size-1
    while curr_length_component >= 1:
        numbits_required = math.ceil(math.log2(curr_length_component+1))
        computed_bitArr = minimal_bin_code(numbits_required, curr_length_component)
        # make the first bit 0
        computed_bitArr[0]=False
        length_components.append(computed_bitArr)
        curr_length_component = numbits_required-1
    return (length_components, code_comp)


def bin_to_dec(bitArr):
    """
    Utility function to convert an array of bits into the decimal number it represents
    """
    dec = 0
    power = len(bitArr)-1
    for bit in bitArr:
        dec += bit * (2**power)
        power -= 1

    return dec


def decode_eliasomega(bitArr):
    readLength = 1
    pos = 1

    component_bin_arr = []
    while True:
        component_bin_arr = []
        for i in range(pos, pos+readLength-1+1):
            component_bin_arr.append(bitArr[i-1])
        
        if component_bin_arr[0] == 1:
            return bin_to_dec(component_bin_arr)
        else:
            component_bin_arr[0] = 1
            pos = pos + readLength
            bin_rep = bin_to_dec(component_bin_arr)
            readLength = bin_rep + 1



def decode_huffman(huffmanTreeRoot, encodedString):
    originalString = ""
    curr = huffmanTreeRoot
    for i in range(len(encodedString)):
        if encodedString[i] == '0':
            curr = curr.left
        else:
            curr = curr.right
 
        # reached leaf node
        if curr.left is None and curr.right is None:
            originalString += curr.string
            curr = huffmanTreeRoot
    return originalString 


def computeRankTable(string, suffix_arr, rank_arr):
    table = [None for i in range(126-36+1)]
    for suffix in suffix_arr:
        # if encountered for the first time, insert rank
        if table[ord(string[suffix-1])-36] is None:
            table[ord(string[suffix-1])-36] = rank_arr[suffix-1]

    return table


def invert_bwt(bwt_str, rank_table):
    outputStr = [None for i in range(len(bwt_str))]
    fill_char_idx = len(bwt_str)-1
    outputStr[fill_char_idx] = '$'
    i = 1
    last_column_char = bwt_str[i-1]
    fill_char_idx -= 1
    outputStr[fill_char_idx] = last_column_char
    fill_char_idx -= 1
    while True:
        last_column_char = bwt_str[i-1]
        if last_column_char == '$':
            break
        # compute occurrences
        counter = 0
        for idx in range(i-1):
            if bwt_str[idx] == last_column_char:
                counter += 1
        pos = rank_table[ord(last_column_char)-36]+counter
        outputStr[fill_char_idx] = bwt_str[pos-1]
        fill_char_idx -= 1
        i = pos
    return outputStr

def encodeString(string, file):
    bitArrays = []

    # encode the Elias Omega codeword for length of BWT strength
    suffix_tree_root = ukkonen(string)
    str_suff_arr = suffix_array(suffix_tree_root, len(string), suffix_tree_root)
    str_bwt = bwt(string, str_suff_arr)
    eo_res = elias_omega(len(str_bwt))

    for i in range(len(eo_res[0])-1, -1,-1):
        bitArrays.append(eo_res[0][i])
    bitArrays.append(eo_res[1])

    # encode number of distinct characters in bwt string
    # define lookup for each unique character in str_bwt
    unique_chars = []
    char_dict = [None] * (126-36+1)
    nunique = 0
    for char in str_bwt:
        ascii_code = ord(char)
        if char_dict[ascii_code-36] is None:
            char_dict[ascii_code-36]= True
            unique_chars.append(ascii_code)
            nunique += 1
    # same as above
    nunique_eo_res = elias_omega(nunique)
    for i in range(len(nunique_eo_res[0])-1,-1,-1):
        bitArrays.append(nunique_eo_res[0][i])
    bitArrays.append(nunique_eo_res[1])

    # compute huffman code for characters based on bwt string
    huff_encodings = huffmanEncoding(str_bwt)

    # compute fixed-width 7 bit ASCII code
    for char_code in unique_chars:
        char_code_copy = char_code
        bitArr = bitarray(7)
        index = 6
        while char_code > 0:
            bitArr[index] = char_code%2
            char_code //=2
            index -= 1
        bitArrays.append(bitArr)
        huff_code = huff_encodings[char_code_copy-36]
        huff_code_length = len(huff_code)
        huff_code_length_eo_encoding = elias_omega(huff_code_length)
        # add the eo encodings of the huffman encoding length
        for i in range(len(huff_code_length_eo_encoding[0])-1, -1,-1):
            bitArrays.append(huff_code_length_eo_encoding[0][i])
        bitArrays.append(huff_code_length_eo_encoding[1])
        # add the huffman encodings 
        bitArrays.append(huff_code)


    # encode the runlength tuples
    
    run_len = 1
    run_len_tuple = (0, run_len)
    str_idx = 1
    while str_idx <= len(str_bwt)-1:
        curr_char = str_bwt[str_idx]
        if curr_char == str_bwt[run_len_tuple[0]]: # if current char same as previous char
            run_len_tuple = (str_idx, run_len_tuple[1]+1)
        else: # mismatch means we can encode the runlength tuple of prev
            char_huff = huff_encodings[ord(str_bwt[run_len_tuple[0]])-36]
            run_len_eo = elias_omega(run_len_tuple[1])
            bitArrays.append(char_huff)
            for i in range(len(run_len_eo[0])-1, -1,-1):
                bitArrays.append(run_len_eo[0][i])
            bitArrays.append(run_len_eo[1])            
            run_len_tuple = (str_idx, 1)
        str_idx += 1
    # perform last update 
    char_huff = huff_encodings[ord(str_bwt[run_len_tuple[0]])-36]
    run_len_eo = elias_omega(run_len_tuple[1])
    bitArrays.append(char_huff)
    for i in range(len(run_len_eo[0])-1, -1,-1):
        bitArrays.append(run_len_eo[0][i])
    bitArrays.append(run_len_eo[1])            


    # now that we have everything, we can write to file
    buffer = bitarray(8)
    # write to file in buffers of size 8
    counter = 0
    for bit_array in bitArrays:
        for bit in bit_array:
            buffer[counter]=(bit)
            counter += 1
            if counter == 8: # send 8 at once
                file.write(buffer)
                buffer = bitarray(8)
                counter = 0

    if counter < 8: # send the remainder
        file.write(buffer)


class Vertex:
    """
    Used for min heap
    """
    def __init__(self, item, string):
        self.string = string
        self.item = item
        self.edges = [] # adjacency list
        self.index = None
        self.left = None
        self.right = None
        self.bit = None
        self.bitArr = None
        self.depth = 0
        self.previous = None

    def __str__(self):
        return str(self.string) + " " + str(self.item) + " " + str(self.bitArr)

############################# Code for generating suffix array used in bwt ##############################
def ukkonen(string):

    # debugging
    nodes = []
    # add terminal character
    n = len(string)

    # init variables, active node, remainder, last_j, globalEnd
    remainder = (0,0)
    lastJ = 0
    globalEnd = [0]
    ruleUsed = None
    first = ord('a') # for naming intermediate nodes

    for i in range(1, n+1):
        globalEnd[0] += 1
        pointer = globalEnd

        # base case
        if i == 1:
            root = Node("r", True)
            firstNode = Node(1, True)
            char = string[0]
            root.edges[ord(char)-36] = Edge(root, firstNode, 1,pointer)
            # debugging
            nodes.append(root)
            nodes.append(firstNode)
            activeNode = root
            lastJ += 1
            root.suffixLink = root
        else:
            # begin phase i+1 and do leaf extension
            j = lastJ+1
            # below three variables are for resolving suffix links
            oldInternalNode = None
            newInternalNode = None
            needToResolve = False 
            while j < i+1:
                if ruleUsed == 3: # if previous extension created internal node, save a copy of that internal node
                    oldInternalNode = newInternalNode
                newInternalNode = None # reset internal node because we don't know if this extension will create a new one
                ruleUsed = None
                # skip count, not needed if remainder = (0,0). 
                if remainder != (0,0):
                    remainderLength = remainder[1]-remainder[0]+1 # amount to traverse
                    counter = remainder[0] # used for checking first character of next edge when skipping
                    currEdge = activeNode.edges[ord(string[remainder[0]-1])-36] # first edge to examine
                    currEdgeLength = (currEdge.label[1][0]-currEdge.label[0])+1 # for the edge to be examined
                    cumulativeLength = 0
                    isRule2Case1 = False # this is to check if we traverse exactly the amount needed on current edge
                    while remainderLength >= (cumulativeLength + currEdgeLength):
                        activeNode = currEdge.tail # go to next node
                        counter += currEdgeLength # update next character to check 
                        currEdge = activeNode.edges[ord(string[counter-1])-36] # this is the next edge to be examined (if it exists)
                        if currEdge is not None: # rule 2 case 1
                            cumulativeLength += currEdgeLength # update total traversed/skipped
                            currEdgeLength = (currEdge.label[1][0]-currEdge.label[0])+1 # update the next edge length to be considered
                        else:
                            isRule2Case1 = True # the next edge is not found, so we need to directly apply rule 2: case 1 extension
                            break
                    if isRule2Case1:
                        traverseAmount = 0
                    else:
                        traverseAmount = remainderLength-cumulativeLength # this is the remaining amount to traverse on the edge where extension occurs

        
                    # perform extension at extension point
                    if traverseAmount != 0: # we are halway through an edge
                        charPosition = currEdge.label[0] + traverseAmount - 1 # this is how we skip the traversal, by computing the char index to be checked
                        edgeChar = string[charPosition]
                        if string[(i-1)] == edgeChar: # check rule 3:
                            ruleUsed = 4
                        else: # rule 2 case 2
                            ruleUsed = 3
                            oldStart = currEdge.label[0] 
                            oldEndNode = currEdge.tail 
                            # create intermediate node inside old edge
                            intermediateNode = Node(chr(first), False)
                            first += 1 # TO BE REMOVED 
                            newInternalNode = intermediateNode # used for resolving suffix links
                            # update old edge to end at this intermediate node      
                            newEnd = oldStart + traverseAmount - 1 # 1
                            newEdge = Edge(activeNode, intermediateNode, oldStart, [newEnd])
                            activeNode.edges[ord(string[oldStart-1])-36] = newEdge
                            # create new edge between intermediate node and new leaf node
                            newCharNode = Node(j, True) # create new leaf node for character 
                            newEdge = Edge(intermediateNode, newCharNode, pointer[0], pointer)
                            intermediateNode.edges[ord(string[pointer[0]-1])-36] = newEdge
                            nodes.append(newCharNode) # TO BE REMOVED
                            # create new edge between intermediate node and old end node
                            remainderStart = currEdge.label[0]+traverseAmount
                            remainderEnd = currEdge.label[1][0]
                            if currEdge.tail.isLeaf: # if original end node, was leaf, it needs to remain a pointer to global end
                                newEdge = Edge(intermediateNode, oldEndNode, remainderStart, pointer)
                            else:
                                newEdge = Edge(intermediateNode, oldEndNode, remainderStart, [remainderEnd])
                            intermediateNode.edges[ord(string[remainderStart-1])-36] = newEdge
                            nodes.append(intermediateNode) # TO BE REMOVED
                            lastJ+=1 # for rule 2 extensions
                            remainder = (oldStart,newEnd) # CHECK IF CAN BE REMOVED
                    else: # we are at a node, and traverse amount is 0
                        # update remainder since no more characters to traverse below active node to reach extension point
                        remainder = (0,0)
                        # check rule 3 
                        if activeNode.edges[ord(string[i-1])-36] is not None:
                            ruleUsed = 4
                        else: # rule 2 case 1
                            ruleUsed = 2
                            newNode = Node(j, True)
                            nodes.append(newNode)
                            activeNode.edges[ord(string[i-1])-36] = Edge(activeNode, newNode, i, pointer)
                            lastJ+=1 # for rule 2 extensions                                                         
                
                else: # remainder is empty, meaning no skip counting; we directly apply extension rules
                    if activeNode.edges[ord(string[i-1])-36] is None:
                        ruleUsed = 2
                        newNode = Node(j, True)
                        nodes.append(newNode)
                        activeNode.edges[ord(string[i-1])-36] = Edge(activeNode, newNode, i, pointer)
                        lastJ+=1
                    else:
                        ruleUsed = 4

               
                if ruleUsed == 4:
                    j = i+1 # terminate this phase
                    # add character str[i] to remainder 
                    if remainder == (0,0):
                        remainder = (i,i)
                    else:
                        remainder = (remainder[0], remainder[1]+1)
                else:
                    j+=1

                # resolve suffix links
                if needToResolve:
                    if newInternalNode is None: # if no internal node created in this extension, suffix link points to active node
                        oldInternalNode.suffixLink = activeNode
                    else:
                        oldInternalNode.suffixLink = newInternalNode # the active node should be the new internal node


                needToResolve = (ruleUsed == 3) # boolean indicating if resolving is needed for next extension 

                if ruleUsed == 2 or ruleUsed == 3:
                    # if active node == root, then activeNode.suffixLink = root, and 
                    # manually remove first char from remainder if active node is root
                    if activeNode == root:
                        if remainder[0] == remainder[1]: 
                            remainder = (0,0)
                        else:
                            remainder = (remainder[0]+1, remainder[1])
                    activeNode = activeNode.suffixLink # rule 2 extensions mean we should follow suffix link to the new activeNode for next extension
    return root

def dfs_suffix_tree(node, array, strLength, root):
    if node is None:
        return
    
    # If the current node is a leaf node, add its value to the visited leaves array
    if node.isLeaf and node != root:
        array.append(node.id)
    
    # Recursively traverse each child
    for edge in node.edges:
        if edge is not None:
            dfs_suffix_tree(edge.tail, array, strLength, root)

    if len(array) == strLength:
        return array

    return array # check if this is ok

def print_tree(nodes):
    for node in nodes:
        print("Node", node.id, ":")
        edges = []
        for edge in node.edges:
            if edge is not None:
                if edge.tail is not None:
                    arg1 = edge.tail.id
                else:
                    arg1 = None
                if edge.head is not None:
                    arg2 = edge.head.id
                else:
                    arg2 = None
                edges.append(str(arg2)+" "+str(edge.label)+" "+str(arg1))

def suffix_array(node, strLength, root):
    array = []
    array = dfs_suffix_tree(node, array, strLength, root)
    return array

class Node:
    CHAR_RANGE = 126-36+1 #(94)

    def __init__(self, id, isLeaf):
        self.id = id
        self.isLeaf = isLeaf
        self.suffixLink = None
        self.edges = [None] * self.CHAR_RANGE

class Edge:

    def __init__(self,head,tail, start,end):
        self.head = head
        self.tail = tail
        self.label = (start,end)

    def __str__(self):
        if self.tail is not None:
            arg1 = self.tail.id
        else:
            arg1 = "None"
        if self.head is not None:
            arg2 = self.head.id
        else:
            arg2 = "None"
        return arg2 + " " + str(self.label[0]) + " " + str(self.label[1][0]) + " " + str(arg1)
    

if __name__ == "__main__":

    strF = open(sys.argv[1], "r")
    string = strF.readline() # read the string from the file
    with open('q2_encoder_output.bin', 'wb') as file:
        encodeString(string, file)