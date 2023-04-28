import argparse

def main():

    # key = str representation, val = function that accepts image path
    func_dict = dict([])

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="image path", required=True)
    parser.add_argument("--func", type=str, help="triangulation function", choices=list(func_dict), required=True)

    args = parser.parse_args()
    path = args.path
    func = func_dict[args.func]

    func(path)

if __name__ == '__main__':
    main()