"""
Name: Khor Jia Wynn
Student ID: 33033919
"""
from bitarray  import bitarray

class HuffmanTreeNode:
    """
    Node for huffman tree
    """
    def __init__(self,item):
        self.item = item
        self.left = None
        self.right = None
        self.isLeaf = None


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


def decode_eliasomega(counter, bitArr):
    """
    Decodes elias omega bitarray. The counter variable helps to specify at which point of the bitArr do we perform the decoding.
    Returns a tuple of two items, first is the next index in the bitArr to perform further decoding, second item is decoded integer
    """
    readLength = 1
    pos = counter + 1

    component_bin_arr = []
    while True:
        component_bin_arr = []
        for i in range(pos, pos+readLength-1+1): # this specifies the component we're looking at
            component_bin_arr.append(bitArr[i-1])
        
        if component_bin_arr[0] == 1: # termination condition
            return (pos+readLength-1, bin_to_dec(component_bin_arr))
        else: 
            component_bin_arr[0] = 1 # set the first bit back to 1
            pos = pos + readLength # update the start index of next component to consider
            dec_rep = bin_to_dec(component_bin_arr) # get the decimal rep
            readLength = dec_rep + 1 # update next read length


def invert_bwt(bwt_str):
    """
    Inverts a bwt to the original string
    """
    first_col = [] # first column
    nOccurrences = [None for i in range(126-36+1)] # stores number of occurrences for a character in lexicographical order
    for char in bwt_str:
        if nOccurrences[ord(char)-36] is None: # create entry
            nOccurrences[ord(char)-36] = 1 
        else:
            nOccurrences[ord(char)-36] += 1 # encountered before, so just update freq
    # recall that first column is suffix array (sorted lexicographically), and nOccurrences has inherent lexicographical ordering,
    # so construct the first column based on each index with its mapped frequency (index + 36 is the ascii code)
    for i in range(len(nOccurrences)): 
        if nOccurrences[i] is not None:
            for j in range(nOccurrences[i]):
                first_col.append(i+36)
    # compute rank table
    rank_table = [None for _ in range(126-36+1)]
    for i in range(len(first_col)):
        if rank_table[first_col[i]-36] is None:
            rank_table[first_col[i]-36] = i+1 # +1 because we use 1-based indexing in our calculations

    outputStr = [None for _ in range(len(bwt_str))] # init array to store characters of the output string 
    fill_char_idx = len(bwt_str)-1 # tracks the index of the character to be inserted, remember we construct backwards
    outputStr[fill_char_idx] = '$'
    i = 1
    last_column_char = bwt_str[i-1]
    fill_char_idx -= 1
    outputStr[fill_char_idx] = last_column_char
    fill_char_idx -= 1
    while True:
        last_column_char = bwt_str[i-1]
        if last_column_char == '$': # termination condition
            break
        # compute occurrences of last column character until i (exclusive)
        counter = 0
        for idx in range(i-1):
            if bwt_str[idx] == last_column_char:
                counter += 1
        pos = rank_table[ord(last_column_char)-36]+counter # formula to compute pos
        outputStr[fill_char_idx] = bwt_str[pos-1] # append 
        fill_char_idx -= 1
        i = pos # update index of next last column char to be used
    return outputStr

    

def decode_string(counter, bit_stream, output_fname):
    counter = 0 # index of current bit to process

    # read first elias omega code for the length of the bwt string
    eo_bwt_len = decode_eliasomega(counter, bit_stream)
    next_idx = eo_bwt_len[0]
    length_str_bwt = eo_bwt_len[1] # used for stopping once we have reconstructed the original string

    # read second elias omega code for the number of unique chars
    eo_nunique = decode_eliasomega(next_idx, bit_stream)
    next_idx = eo_nunique[0]
    nunique = eo_nunique[1]

    # loop nunique char times
    times = 0
    # accumulate mapping of ascii code dec with huffman code
    huffcode_mapping = []
    while times < nunique:
        # read 7 bits to get ascii code
        ascii_code_dec = 0
        power = 6
        for i in range(power, -1, -1):
            if bit_stream[next_idx]:
                ascii_code_dec += 2**i
            next_idx += 1
        # read elias code for the char's huffman code length
        eo_huffcode_len = decode_eliasomega(next_idx, bit_stream)
        next_idx = eo_huffcode_len[0]
        huffcode_len = eo_huffcode_len[1]
        # read huffcode_len bits to get huffman code for this char
        huffcode_bits = []
        for i in range(huffcode_len):
            huffcode_bits.append(bit_stream[next_idx])
            next_idx += 1
        times += 1
        huffcode_mapping.append((huffcode_bits, ascii_code_dec))
 
    # construct huffman code tree based on the huffman codes for each unique char
    root = HuffmanTreeNode(None)
    for tup in huffcode_mapping:
        curr_node = root
        for bit in tup[0]:
            if bit == 0:
                if curr_node.left is None:
                    curr_node.left = HuffmanTreeNode(None)
                curr_node = curr_node.left
            elif bit == 1:
                if curr_node.right is None:
                    curr_node.right = HuffmanTreeNode(None)
                curr_node = curr_node.right
        curr_node.isLeaf = True
        curr_node.item = tup[1]

    # starting from huffman tree root, traverse according to the bits in the bitstream
    # if we ever reach a leaf node, stop traversing, decode elias omega to get length, then append char length times. repeat process
    original_bwt = []
    stop_counter = 0
    while next_idx <= len(bit_stream)-1:
        # traverse tree
        curr = root 
        character = None
        if stop_counter == length_str_bwt: # stop if we have the full string
            break
        while True:
            curr_bit = bit_stream[next_idx]
            if curr_bit == 0:
                curr = curr.left
            else:
                curr = curr.right
            next_idx += 1
            if curr.isLeaf: # leaf node reached, this is the character
                character = chr(curr.item)
                break

        if stop_counter == length_str_bwt: # stop if we have the full string
            break
        # decode elias omega for number of times this character is outputted
        eo_char_occurrences = decode_eliasomega(next_idx, bit_stream)
        next_idx = eo_char_occurrences[0]
        char_occurrences = eo_char_occurrences[1]
        for i in range(char_occurrences):
            stop_counter += 1
            original_bwt.append(character)
 
    # get original string then write it to file
    original_str = invert_bwt(original_bwt)
    with open(output_fname, "w") as file:
        for char in original_str:
            file.write(char)

if __name__ == "__main__":

    output_fname = "q2_decoder_output.txt"
    # Open the binary file
    with open('q2_encoder_output.bin', 'rb') as file:
        # Read binary data from the file
        binary_data = file.read()

        # Create a bitarray stream from the binary data
        bit_stream = bitarray()
        bit_stream.frombytes(binary_data)

        # Process the bit stream and decode
        counter = 0 # processing starts with first character in bitstream
        decode_string(counter, bit_stream, output_fname)

