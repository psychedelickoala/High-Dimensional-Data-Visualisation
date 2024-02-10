# High-Dimensional-Data-Visualisation
Representing high dimensional data in two dimensions.

Navigate to:
- [Setting up](https://github.com/psychedelickoala/High-Dimensional-Data-Visualisation#setting-up)
- [Adding arguments](https://github.com/psychedelickoala/High-Dimensional-Data-Visualisation#adding-arguments)
- [Valid CSV formats](https://github.com/psychedelickoala/High-Dimensional-Data-Visualisation#valid-csv-formats)
- [Interactive controls](https://github.com/psychedelickoala/High-Dimensional-Data-Visualisation#interactive-controls)

## Setting up
Download hddv.zip (roughly 70 MB) and double click to extract the hddv folder. This folder contains hddv.exe, a samples folder of complimentary data to try, and a _internal folder, which is necessary to the running of hddv.exe. It may be helpful to save your data into this folder to shorten the relative path to these files.

## Adding arguments
Due to its argparse functionality, hddv.exe must be run from the terminal. Open the terminal (Mac) or command line (Windows) and navigate to the location of hddv.exe. 

A basic usage constitutes:
```
./hddv path-to-data-file.csv
```
For example:
```
Annalisas-MBP:~ annalisacalvi$ cd Programs/High-Dimensional-Data-Visualisation/hddv
Annalisas-MBP:High-Dimensional-Data-Visualisation annalisacalvi$ ./hddv /Users/annalisacalvi/Programs/
High-Dimensional-Data-Visualisation/samples/spaced_clusters.csv
Finding projections...: 100%|█████████████████████████████████████████████████████████████| 20/20 [00:23<00:00,  1.17s/it]
```
Note that, the first time this is run, it can take about 30 seconds to a minute to start up. After the "Finding projections" bar reaches 100%, a widget should open, such as
![Spaced clusters widget](./images/sc_home.png)
It is recommended to use this app in fullscreen. However, you should still keep an eye on the terminal; some functions will print information there. 

If hddv.exe shares a directory with the data file, you can use the shared path, eg.
```
./hddv samples/spaced_clusters.csv
```
for the same result. Further demonstrations will use relative paths.

There are a number of optional parameters you can use to customise the widget. For a summary, use the command
```
./hddv --help
```
This will print:
```
usage: hddv [-h] [-sp SHARED_PATH] [-d DEPENDENT_DATA]
            [-ci COV_INDEPENDENT_DATA] [-cd COV_DEPENDENT_DATA]
            [-mi MEAN_INDEPENDENT_DATA] [-md MEAN_DEPENDENT_DATA] [-sc] [-scd]
            data

positional arguments:
  data                  path to independent data

options:
  -h, --help            show this help message and exit
  -sp SHARED_PATH, --shared_path SHARED_PATH
                        shared path to all entries, ending with '/'
  -d DEPENDENT_DATA, --dependent_data DEPENDENT_DATA
                        path to dependent data
  -ci COV_INDEPENDENT_DATA, --cov_independent_data COV_INDEPENDENT_DATA
                        path to covariance matrix for independent data
  -cd COV_DEPENDENT_DATA, --cov_dependent_data COV_DEPENDENT_DATA
                        path to covariance matrix for dependent data
  -mi MEAN_INDEPENDENT_DATA, --mean_independent_data MEAN_INDEPENDENT_DATA
                        path to ellipse centre for independent data
  -md MEAN_DEPENDENT_DATA, --mean_dependent_data MEAN_DEPENDENT_DATA
                        path to ellipse centre for dependent data
  -sc, --scale          scale data so each dimension has standard deviation 1
  -scd, --scale_dependent
                        scale dependent data so each dimension has standard
                        deviation 1
```

These methods are explained in further detail below.

### Specifying a covariance matrix
When only one dataset is being analysed, only tags mentioning independent data are relevant. We use the optional parameter ```-ci``` to specify a data file for the covariance matrix.

Syntax:
```
./hddv path-to-data-file.csv -ci path-to-covariance-matrix.csv
```

For example, we could run:
```
./hddv samples/p5pexample/np.csv -ci samples/p5pexample/cov_mat.csv
```

When we are using several data files in the same folder, we can shorten this command by using the ```-sp```, or ```--shared_path```, option:

```
./hddv -sp samples/p5pexample/ np.csv -ci cov_mat.csv
```

This reads ```np.csv``` as our data, ```cov_mat.csv``` as our covariance matrix, and pastes ```samples/p5pexample/``` in front of both these names when reading from the files.

### Specifying the centre

Just like the covariance matrix, we use an optional parameter ```-mi``` to specify the centre of our ellipses.

Syntax:
```
./hddv path-to-data-file.csv -mi path-to-centre.csv
```

We can include as many optional parameters as we like. For example, we can specify a covariance matrix and a centroid, using a shared path as follows:
```
./hddv -sp samples/p5pexample/ np.csv -ci cov_mat.csv -mi centre_of_ellipses.csv
```

### Adding a dependent set of data

You can view two linked datasets side by side, as long as they have the same number of points. This can be useful for analysing functions. In the widget, one dataset (specified as independent) can be manoeuvred freely using lasso select and clustering, and the dataset specified as dependent will colour its linked points accordingly. Two points are 'linked' if they have the same position (disregarding headers) in their respective CSV files. For more information about required data formats, see [valid CSV formats](https://github.com/psychedelickoala/High-Dimensional-Data-Visualisation#valid-csv-formats).

Dependent data is specified using the optional parameter ```-d```. For example:
```
./hddv -sp samples/bphys/ testinput.csv -d testoutput.csv
```
This yields a different looking widget:
![Function widget](./images/function.png)

The plot on the left shows the independent data, and the plot on the right shows the dependent data. 
A covariance matrix and centre can be specified for dependent data as well, using the options ```-cd``` and ```-md``` respectively. These have no effect if a dependent dataset is not specified.

### Standardising data
You may or may not want to standardise your data before plotting it. Choosing to standardise your data means all your dimensions will be scaled to have a standard deviation of 1. Flag with ```-sc``` to scale independent data and ```-scd``` to scale dependent data, ie.

```
./hddv -sp samples/ v1.csv -d v2.csv -sc -scd
```

## Valid CSV formats

All CSV files should be comma-delimited.

### For datasets

Datasets may have one header line. Headers should not be numbers, or the header line will be read as an additional datapoint. It is also recommended that headers are short, say 1-4 letters, as they are used to label the axes on the plot.

Any additional columns, such as ids for the points or a clustering index, will show up as additional dimensions, and should not be present. Each column should represent a parameter and each row (except the optional header) should represent a point in that parameter space.

For linked datasets, the independent and dependent data must be in two separate files, and have the same number of lines minus headers. Order is important; points are linked to each other by their line number (minus headers).

### For the covariance matrix

A $p\times p$ covariance matrix should have exactly $p$ columns and $p$ rows, with no headers, identifiers, or other additional information. The matrix itself must be symmetric and positive-definite.

### For the centre

A $p$ dimensional centre consists of $p$ numbers. It is equally valid for this to be in one row or one column. Again, no additional information should be present.


## Interactive controls

Once you've opened the widget, there are many ways to interact. All the standard ```matplotlib``` controls are present; see [here](https://matplotlib.org/3.2.2/users/navigation_toolbar.html) for more details. Additionally, there are the following keyboard controls:

| Key | Action      |
| --- | ----------- |
| M   | Print projection matrix to terminal       |
| N   | Print CSV file line numbers of **selected** points to terminal       |
| R   | Show random plane (also zooms out if you have zoomed in)       |
| V   | Toggle auto-clustering distance metric between Euclidean, Mahalanobis and Standardised        |
| Z   | Clears all clusters        |

There are also three buttons, corresponding to the actions:

| Button | Action      |
| --- | ----------- |
| Print info  | Print information about the controls to the terminal       |
| Lasso   | Lasso points to manouvre projection or add cluster, see lasso controls       |
| Auto cluster   | Automatically group **selected** points into clusters       |

After lassoing some points, the user is prompted to "Press enter to change projection, Y to add cluster". Specifically, the available controls are:

| Key | Action      |
| --- | ----------- |
| "Y"  | Make a new cluster from lassoed points. Does not select points or change projection.       |
| Enter   | Select lassoed points and optimise projection to best show these points. Does not make a new cluster.       |
| Esc   | Exit lasso tool without changes.       |

After making a new cluster, a coloured rectangle will show up in the empty green rectangle (bottom left in InteractiveGraph, bottom right in InteractiveFunction). Clicking on this rectangle will interact with the corresponding cluster as follows:

| Click | Action      |
| --- | ----------- |
| Right click | Select points in cluster, and optimise projection to best show only that cluster.       |
| Double click   | Delete cluster.       |

In both applications, there is a slider. Changing this slider will change:
- The selected points, which become those (high-dimensional) points with a standard deviation *greater* than the slider value
- The projection, which is optimised to best show the selected points

There are, lastly, checkboxes corresponding to the ellipses. There are six checkboxes:

| Label | Action      |
| --- | ----------- |
| 1σ - 8σ (generally $n$-σ) | Show the 2D projection of the high-dimensional ellipsoid corresponding to $n$ standard deviations from the mean   |
| slider   | Override other checkboxes; match the ellipse standard deviations to the slider value   |


Let me know in the discussions if you have any questions regarding this project!

