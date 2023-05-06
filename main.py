import argparse
from implementations.normal import display_mosaic_baseline, display_mosaic_w_edges
from implementations.constrained import display_mosaic_w_constrained_edges

def main():

    # key = str representation, val = function that accepts image path + arguments
    func_dict = dict({"standard": display_mosaic_baseline, 
                      "edge_pts": display_mosaic_w_edges, 
                      "constrained": display_mosaic_w_constrained_edges})

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="image path", required=True)
    parser.add_argument("--func", type=str, help="triangulation function", choices=list(func_dict), required=True)
    parser.add_argument("--sigma_x", type=int, help="x variance for image blurring")
    parser.add_argument("--sigma_y", type=int, help="y variance for image blurring")
    parser.add_argument("--num_points", type=int, help="number of points for normal triangulation")
    parser.add_argument("--num_edge_points", type=int, help="number of edge points for triangulation")
    parser.add_argument("--min_length", type=int, help="minimum length for edges in constrained triangulation")
    parser.add_argument("--theta_thresh", type=float, help="minimum angle for pairs of segments")
    parser.add_argument("--min_dist", type=int, help="minimum distance between points")
    parser.add_argument("--num_add_points", type=int, help="number of addonal points for triangulation")

    args = parser.parse_args()
    path = args.path
    func = func_dict[args.func]

    func(path, vars(args))

if __name__ == '__main__':
    main()