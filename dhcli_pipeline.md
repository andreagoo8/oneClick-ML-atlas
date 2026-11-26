# oneClick-ML-dhcli

## Install the dhcli utility
Follow the instructions on [this page](https://scc-digitalhub.github.io/docs/0.16/cli/installation/) to install the utility on your operating system.

## Preliminary operations

Open the main repository folder in the terminal. Running `ls` should return

```bash
$ ls
dhcli  dhcli_pipeline.md  main.ipynb  oneClick-ML  README.md
```

Register your instance's configuration by running these commands (more details about them on the [documentation page](https://scc-digitalhub.github.io/docs/0.16/cli/usage/)):

```bash
export DHCORE_ENDPOINT='https://core.dsh.atlas.fbk.eu'
dhcli register -e dsh-platform "$DHCORE_ENDPOINT"
dhcli login -e dsh-platform
```

Create the project and check the success of the operation; to do so, it’s convenient to create a variable containing the project name that can be also used by other commands (remember to place the actual name in place of `<project_name>` ).

```bash
# Creates a variable with the chosen project name
export PROJECT_NAME='<project_name>'

# Creates the project with the specified name
dhcli create projects -n "$PROJECT_NAME"

# Checks if the project exists
dhcli get projects "$PROJECT_NAME"
```

## Upload or log the artifacts

### Only for the first run

A file that is ready to be used, in the Digitalhub terminology, is an artifact. After that, it can be found on the "Data items" and "Artifact" section of the Core managment UI.

If these input files are already registered and you want to reuse them, skip the uploads and go to the next section.

The two files in this example are `data.csv` and `config.ini`. The following commands shows how to load them in the platform.

First load the csv data file:

```bash
# Uplaod the artifact
dhcli upload -p "$PROJECT_NAME" artifact \
  -n csv_input_data -f oneClick-ML/data.csv
```

Then load the ini config file:

```bash
# Uplaod the artifact
dhcli upload -p "$PROJECT_NAME" artifact \
  -n config_ini_file -f oneClick-ML/config.ini
```

### All the runs

To get a list of all the versions of an artifact, execute `dhcli list -p "$PROJECT_NAME" artifacts -n <artifact_name>`

```bash
dhcli list -p "$PROJECT_NAME" artifacts -n csv_input_data
dhcli list -p "$PROJECT_NAME" artifacts -n config_ini_file
```

Retrieve the files and assign a convenient variable to their corresponding key. Running simply `dhcli get -p "$PROJECT_NAME" artifacts -n <artifact_name>` gets the latest version, while `dhcli get -p "$PROJECT_NAME" artifacts <artifact_ID>` gets the specified version.

```bash
# Retrieve the files
dhcli get -p "$PROJECT_NAME" artifacts -n csv_input_data
dhcli get -p "$PROJECT_NAME" artifacts -n config_ini_file
```

To assign the key of latest version of the artifacts to variables (it will be convenient later), run these commands:

```bash
# Assign the keys to the corresponding variables
export CSV_KEY=$(dhcli get -p "$PROJECT_NAME" artifacts -n csv_input_data | awk '/^Key:/ {print $2}')
export CONFIG_KEY=$(dhcli get -p "$PROJECT_NAME" artifacts -n csv_input_data | awk '/^Key:/ {print $2}')
```

## Function declaration

### Only for the first run

A zip file with all the files is already present in the repo folder, in the path `dhcli/function_source.zip`. In case of a change in the code source, to create it run:

```bash
(
  cd oneClick-ML
  zip -r ../dhcli/oneclick-source.zip main.py params.yaml requirements.txt lib
  if [ -d feature_list_types ]; then
    zip -r ../dhcli/oneclick-source.zip feature_list_types
  fi
)
```

To declare the function and make it usable on the platform, it first needs to be uploaded. Do it with

```bash
# Upload the zip
dhcli upload -p "$PROJECT_NAME" artifact \
  -n function_source -f dhcli/function_source.zip
```

Then check that the upload run successfully with the command

```bash
dhcli get -p "$PROJECT_NAME" -o yaml artifacts -n function_source
```

Its path on the S3 datalake can be found on the “spec > path” entry; conveniently place this path in the `SOURCE_PATH` variable by running the following command.

```bash
export SOURCE_PATH=$(dhcli get -p "$PROJECT_NAME" -o yaml artifacts -n function_source | awk '/^  path:/ {print $2}')
```

The following block validates that `PROJECT_NAME` and `SOURCE_PATH` are set and that it’s run from the expected parent directory, then creates a DhCLI function YAML using the zipped source and dependencies from `oneClick-ML/requirements.txt`. It creates the `main_function` and then fetches its details.

```bash
(
set -e
: "${PROJECT_NAME:?}"
: "${SOURCE_PATH:?}"
if [ ! -r oneClick-ML/requirements.txt ]; then
  printf '%s\n' 'Run this block from the parent of oneClick-ML.' >&2
  exit 1
fi
mkdir -p dhcli

cat > dhcli/main-function.yaml <<YAML
kind: python
name: main_function
project: "$PROJECT_NAME"
spec:
  python_version: PYTHON3_10
  source:
    lang: python
    source: "zip+$SOURCE_PATH"
    handler: main:helper_main
  requirements:
YAML

while IFS= read -r requirement || [ -n "$requirement" ]; do
  case "$requirement" in
    ''|\#*) continue ;;
  esac
  printf '    - "%s"\n' "$requirement" >> dhcli/main-function.yaml
done < oneClick-ML/requirements.txt

dhcli create -p "$PROJECT_NAME" functions -f dhcli/main-function.yaml
dhcli get -p "$PROJECT_NAME" functions -n main_function
)
```

### All the runs

Check that the function is correctly registered in the platform with

```bash
dhcli get -p "$PROJECT_NAME" functions -n main_function
```

To conveniently store the ID of the function in a variable (useful for the later steps), run the following command:

```bash
export FUNCTION_ID=$(dhcli get -p "$PROJECT_NAME" functions -n main_function | awk '/^ID:/ {print $2}')
```

## Function building

Create a yaml file in the path `dhcli/build.yaml` with the notebook's build resources:

```bash
cat > dhcli/build.yaml <<'YAML'
spec:
  resources:
    cpu: "4"
    mem: "8Gi"
    disk: "10Gi"
YAML
```

Then start the building with

```bash
# Starts to build the fuction from the created YAML
dhcli run -p "$PROJECT_NAME" -i "$FUNCTION_ID" \
  -f dhcli/build.yaml python+build
```

Check the progress of the building until it reports “COMPLETED”, or inspect the logs by running the commands:

```bash
# Check until the STATE is completed
dhcli list -p "$PROJECT_NAME" runs

# Check the logs
dhcli log -p "$PROJECT_NAME" -f "<build-run-id>"
```

## Function running

Create a yaml file in the path `dhcli/job.yaml` with the keys of the input files earlier created.

```bash
cat > dhcli/job.yaml <<YAML
spec:
  inputs:
    csv_file: "$CSV_KEY"
    config_file: "$CONFIG_KEY"
  parameters:
    output_artifact_name: output_data
  resources:
    cpu: "4"
    mem: "8Gi"
    disk: "20Gi"
  local_execution: false
YAML
```

Then start the function running with

```bash
# Starts to build the fuction from the created YAML
dhcli run -p "$PROJECT_NAME" -i "$FUNCTION_ID" \
  -f dhcli/job.yaml python+job
```

Check the progress of the running until it reports “COMPLETED”, or inspect the logs by running the commands:

```bash
# Check until the STATE is completed
dhcli list -p "$PROJECT_NAME" runs

# Check the logs
dhcli log -p "$PROJECT_NAME" -f "<run-id>"
```
