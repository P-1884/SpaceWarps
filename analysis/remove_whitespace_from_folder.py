import sys
args =sys.argv
assert len(args) ==2 #Argument to the code should be path name of file to remove white-space from
original_file = args[1]
orig = open(original_file,'r')

temporary_file = '/Users/hollowayp/vics82_swap_pjm_updated/analysis/newtestfilecandelete.txt'
temp = open(temporary_file,'w')

for line in orig:
    print(line)
    temp.write(" ".join(line.split())+'\n')

orig.close()
temp.close()

#Now overwriting the original:
orig = open(original_file,'w')
temp = open(temporary_file,'r')

for line in temp:
    orig.write(line)

orig.close()
temp.close()