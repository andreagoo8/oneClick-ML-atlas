# How to use the oneClick-ML utility with the ATLAS platform
Create a project in the Core management UI (let's suppose to name it `oneClick-ML-atlas`).

Create a workspace with the Coder utility and start it. Open a terminal into the mother folder and clone the repository with `git clone <repo_address>`, or alternatively upload the archive and extract it with `tar -xzf oneClick-ML-atlas.tar.gz`.

Open the folder `oneClick-ML`.

Upload the csv input file data in this folder.

Set the desired parameters and configurations in the `config.ini` file; pay attention to the `y_label` parameter and make sure it matches the proper column name of the csv file.

Open the `main.ipynb` notebook and set the right project name in the second cell.

Finally, run the cells following the explanations. The execution can be monitored in the "Runs" section of the Core.

The results can be found in the "Artifacts" section.
