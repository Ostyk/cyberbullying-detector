import os

def main(output_name = 'test.md'):
    print("output at: {}".format(output_name))
    tmp = 'test.txt'
    os.system("tree -o test.txt")
    
    with open(tmp) as f:
        lines = f.readlines()

    with open(output_name, 'w') as q:
        for line in lines[1:-2]:
            line = line.rsplit("\n")[0]
            if not line.endswith('.pyc'):
                #print(line.split(" "))
                split = line.split(" ")
                name = split[-1]
                if name != '__pycache__':
                    length = len(split)-1
                    indent = "*" if length == 1 else " " * (length+2) + "*"
                    fmt = "{} [{}]({})\n".format(indent, name, name)
                    #print(fmt)
                    q.write(fmt)
                
if __name__ == '__main__':
    from sys import argv

    if len(argv)==1:
        main()
    else:
        main(argv[1])