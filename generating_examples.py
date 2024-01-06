import numpy as np
import csv
from calculator import Calculator

print("Hello and welcome to my primitive example generator!")
import_data = input("Would you like to use data from sample_data.csv? Y/N: ")
if import_data == "Y":
    print("Importing data...")
    calc = Calculator("sample_data.csv")
    with open("sample_data.csv") as f:
        reader = csv.reader(f,delimiter=",")
        dim = len(next(reader)) # Read first line and count columns
        f.seek(0)
        num_points = sum(1 for _ in reader)
    print("You have imported data from sample_data.csv.")
    print(f"This data has {num_points} points in {dim} dimensions.")

else:
    print("You have selected to specify your own parameters.\n")
    # specify number of dimensions
    print("~~ CHOOSING DIMENSIONS AND NUMBER OF POINTS ~~")
    dim = int(input("Specify a number of dimensions to work in: "))

    # specify number of points in main body of data

    num_data = int(input("Specify the number of data points in the main body of data (eg. 1000): "))
    # SPECIFY COVARIANCE

    # can choose "identity" to use identity matrix, 
    print("\n ~~ CHOOSING COVARIANCE AND CENTRE ~~")
    print("Please specify a method of choosing the covarianceN matrix.")
    print(f"Input I to use the {dim}x{dim} identity matrix.")
    print(f"Input R to use a randomly generated {dim}x{dim} covariance matrix.")
    print(f"Input M to manually enter a {dim}x{dim} covariance matrix.")
    cov_choice = input("Your choice: ")


    if cov_choice == "I":
        cov = np.identity(dim)
    elif cov_choice == "R":
        base = np.random.rand(dim, dim)*0.5
        cov = (base @ base.T) + np.identity(dim)
    elif cov_choice == "M":
        print("\n ~~Manual covariance matrix selection~~")
        print("Please enter your matrix row by row, separating items in the same row with spaces.")
        print("Example input in two dimensions: \n1.3 0.2\n0.2 1.5")
        print("Enter your matrix below:")
        cov = np.float_([input().split() for _ in range(dim)])
        print("Manual covariance entry complete.")
    else:
        print(f"\'{cov_choice}\' is not a valid input.")


    print(f"The specified covariance matrix is: \n{cov}")


    # specify centre
    print("Please specify a method of choosing the centre.")
    print(f"Input O to use the origin.")
    print(f"Input R to use a randomly generated centre.")
    print(f"Input M to manually enter a {dim}x1 centre.")
    centre_choice = input("Your choice: ")

    if centre_choice == "O" or centre_choice == "0":
        centre = np.zeros(dim)
    elif centre_choice == "R":
        centre = 2*np.random.rand(dim) - 1
    elif centre_choice == "M":
        print("\n ~~Manual centre selection~~")
        print(f"Please enter {dim} values, separated by spaces.")
        centre = np.float_(input().split())
        print("Manual centre entry complete.")
    else:
        print(f"\'{centre_choice}\' is not a valid input.")

    print(f"The specified centre is: \n{centre}")

    print("\n~~ INTRODUCING CLUSTERS ~~")
    num_clusters = int(input("Specify a number of additional clusters to add (can be 0): "))

    if num_clusters > 0:
        print("Specify the number of data points in each cluster.")
        print("Enter one value," + ("" if num_clusters==1 else f" or {num_clusters} values separated by spaces, below."))
        cluster_sizes = input().split()
        if len(cluster_sizes) == 1:
            cluster_sizes = [cluster_sizes[0]]*(num_clusters)
        cluster_sizes = np.int_(cluster_sizes)
        
        print("\nSpecify the distance of each cluster from the mean.")
        print("Enter one value," + ("" if num_clusters==1 else f" or {num_clusters} values separated by spaces, below."))
        cluster_dists = input().split()
        if len(cluster_dists) == 1:
            cluster_dists = [cluster_dists[0]]*(num_clusters)
        cluster_dists = np.float_(cluster_dists)

    print("\nGenerating data...")

    data = np.random.multivariate_normal(centre, cov, size=num_data)
    for i in range(num_clusters):
        cluster_direction = 2*np.random.rand(dim) - 1
        cluster_mean = centre + cluster_dists[i]*(cluster_direction/np.linalg.norm(cluster_direction))
        data = np.vstack([data, np.random.multivariate_normal(cluster_mean, cov, size=cluster_sizes[i])])

    print("Data generated according to specifications.")
    write_to_csv = input("\nWould you like to write this data to sample_data.csv? This will overwrite any existing data in the file. Y/N: ")
    if write_to_csv == "Y":
        np.savetxt("sample_data.csv", data, delimiter=",")
        print("Data written to sample_data.csv.")

    calc = Calculator(data.T)


#print("Plotting data...")

#calc.plot_comparison(optimise_cutoffs=[0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12], across = 3, down = 3, verbose = True)
#print("Plotting complete.")
