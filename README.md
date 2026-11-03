# How to use the oneClick-ML utility with the ATLAS platform
Create a project in the Core management UI (let's suppose to name it `oneClick-ML-atlas`).

Create a workspace with the Coder utility that uses Python 3.10, and start it. Open a terminal into the mother folder and clone the repository with `git clone https://github.com/andreagoo8/oneClick-ML-atlas.git`.

Open the folder `oneClick-ML`.

Upload the csv input file data in this folder, and make sure that it's named `data.csv`.

Set the desired parameters and configurations in the `config.ini` file; pay attention to the `y_label` parameter and make sure it matches the proper column name of the csv file.

Open the `main.ipynb` notebook and set the right project name in the cell containing `get_or_create_project()`.

Finally, run the cells while reading the comments. The execution can be monitored in the "Runs" section of the Core.

The results can be found in the "Models" or in the "Files" section (under the "models" folder).
