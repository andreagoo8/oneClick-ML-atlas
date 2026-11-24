# How to use oneClick-ML on the Atlas cluster with `dhcli`

This guide translates `main.ipynb` into terminal commands: create or retrieve the
project, upload the CSV and configuration, register `main:helper_main`, build its
image, and submit a training job. It keeps the notebook's names, Python 3.10, and
resource settings. Run the steps in order in the same Bash terminal, from the
repository root (the directory containing `main.ipynb`). The commands submit work
to Digitalhub; training happens on the cluster.

If your terminal is currently inside `oneClick-ML`, run `cd ..` first. All paths
below are relative to its parent directory, even if this guide is stored inside
`oneClick-ML`.

## 1. Prepare the CLI and the input files

Install a `dhcli` release compatible with Digitalhub 0.15 for your operating system
and architecture, extract it, and put the executable on `PATH`. See the official
[installation instructions](https://scc-digitalhub.github.io/docs/0.15/cli/installation/)
and [release downloads](https://github.com/scc-digitalhub/digitalhub-cli/releases).
The commands below use Bash and `dhcli`. The ML code is already implemented;
there is no local Python orchestration or JSON parser to install. Function and
run settings are supplied as YAML files. Dependencies are installed in the
cluster image during the build.

```bash
dhcli --version
mkdir -p dhcli
export PROJECT_NAME='oneclick-ml-atlas'
```

Prepare `oneClick-ML/data.csv` and edit `oneClick-ML/config.ini` before uploading:

- Set `y_label` to the target column in your CSV and review `col_to_drop`,
  classifiers, cross-validation, and other analysis settings.
- Keep the CSV filename `data.csv` and `input_path = local_data/data/data.csv`.
  The helper downloads the CSV into `local_data/data`.
- Keep `output_path` beneath `local_data/results/`, for example the existing
  `local_data/results/no_active_at_bl/`. The helper uploads that results tree.
- The supplied `path_to_feature_types = ../data/clusters/feature_list_types`
  points outside this repository. If you use feature lists, copy their CSVs into
  `oneClick-ML/feature_list_types/` and set
  `path_to_feature_types = feature_list_types`. Expected names are
  `nominal_features.csv`, `numerical_features.csv`, `ordinal_features.csv`, and
  `binary_features.csv`. These files must travel with the source; workspace paths
  are not automatically available in a job. The loader warns and applies its
  fallback when lists are absent, so check that this behavior suits your data.

`params.yaml` is also required: `main.py` opens it using a relative path. The
code archive used in step 4 must contain it next to `main.py` and `lib/`,
at its root.

## 2. Connect to Digitalhub and create or retrieve the project

Replace the endpoint placeholder with the **Core API base URL** for Atlas,
available from your platform administrator. Register it once, select the
environment, then authenticate. Use the configured environment directly if it
already exists. This follows the documented
[CLI connection flow](https://scc-digitalhub.github.io/docs/0.15/cli/usage/).

```bash
export ATLAS_CORE_ENDPOINT='https://<your-atlas-core-endpoint>'
dhcli register -e atlas "$ATLAS_CORE_ENDPOINT"
dhcli use atlas
dhcli login -e atlas
dhcli list-env
```

The CLI stores its connection configuration in `.dhcore.ini`. Uploads and
downloads also require access to the platform's S3-compatible storage, including
the correct endpoint and credentials for your installation. If Core access works
but transfer fails, check the environment's storage configuration with your
administrator. Do not copy workspace-only filesystem paths into remote sources.

Retrieve the project, corresponding to `dh.get_or_create_project(...)`:

```bash
dhcli get projects "$PROJECT_NAME"
```

If the project does not exist, create it, then retrieve it again. If retrieval
fails because of authentication, connectivity, or permissions, resolve that
error first.

```bash
dhcli create projects -n "$PROJECT_NAME"
dhcli get projects "$PROJECT_NAME"
```

Use the exact project name shown in Core if reusing one: the notebook uses
`oneclick-ml-atlas`, while the README uses different capitalization.

## 3. Load or retrieve the input dataset and configuration

An artifact combines a catalog entry with stored files. These uploads replace
the notebook's two `project.log_artifact(...)` calls. The transfer command
uploads the bytes as well as registering the artifact; creating metadata alone
would not make a local file accessible to the cluster. See
[artifact transfers](https://scc-digitalhub.github.io/docs/0.15/cli/usage/#uploaddownload-a-filefolder).

Run these commands for new inputs:

```bash
dhcli upload -p "$PROJECT_NAME" artifact \
  -n csv_input_data -f oneClick-ML/data.csv
dhcli upload -p "$PROJECT_NAME" artifact \
  -n config_ini_file -f oneClick-ML/config.ini
```

If these inputs are already registered and you want to reuse them, skip the
uploads. In either case, retrieve them, corresponding to
`project.get_artifact(...)`:

```bash
dhcli get -p "$PROJECT_NAME" artifacts -n csv_input_data
dhcli get -p "$PROJECT_NAME" artifacts -n config_ini_file
```

Both artifacts must show `State: READY`. Copy the `Key:` value printed by each
command into these variables, replacing the placeholders. This is the equivalent
of reading `csv_data.key` and `config_data.key` in the notebook:

```bash
export CSV_KEY='<Key printed for csv_input_data>'
export CONFIG_KEY='<Key printed for config_ini_file>'
```

Keys have the form `artifact://oneclick-ml-atlas/csv_input_data:<id>`. They identify
catalog objects; S3 paths identify stored files. The job's `inputs` need entity
keys. To select an older version, list versions and replace `-n name` in the
`get` commands with its ID, then copy that version's key:

```bash
dhcli list -p "$PROJECT_NAME" artifacts -n csv_input_data
dhcli list -p "$PROJECT_NAME" artifacts -n config_ini_file
```

## 4. Declare or retrieve the main helper function

A Digitalhub function describes reusable executable code. Registration, image
building, and execution are separate operations. The handler is the existing
`helper_main` in `oneClick-ML/main.py`. It receives the project automatically,
downloads the input artifacts, calls the ML program, and logs
`local_data/results` as an output artifact. None of that code needs rewriting.

The notebook's short `project.new_function(...)` call also prepares the local
files: the SDK archives and uploads the `code_src` directory and resolves the
requirements file into the function definition. The directory-upload behavior
is implemented in the SDK runtime's
[source preparation code](https://github.com/scc-digitalhub/digitalhub-sdk-runtime-python/blob/0.15.0/digitalhub_runtime_python/entities/function/python/utils.py).
By comparison, the CLI's
[create implementation](https://github.com/scc-digitalhub/digitalhub-cli-sdk/blob/v0.2.4/sdk/services/crud/create.go)
reads a YAML definition and sends it to Core; it does not perform that local
source preparation. The longer procedure below exposes that preparation for a
first registration using only the CLI. It is not required for each execution.

**If `main_function` is already registered with the correct source and
dependencies**, skip source upload and declaration below. Retrieve it:

```bash
dhcli get -p "$PROJECT_NAME" functions -n main_function
```

Copy its printed `ID:` into `FUNCTION_ID`:

```bash
export FUNCTION_ID='<function-id>'
```

Use this same ID for building and running. If you want a specific version, use
its ID instead of `-n main_function`. Continue with step 5, or step 6 if this
version already has a built image.

You can inspect or save the definition that the notebook already registered,
including its stored source reference and resolved dependencies:

```bash
dhcli get -p "$PROJECT_NAME" -o yaml functions "$FUNCTION_ID" \
  > dhcli/main-function.yaml
```

This avoids repackaging the code or reconstructing the requirements when you
continue using that registered function. A local source folder alone is different
from a function already registered in Core.

<details>
<summary>First registration from local files using only dhcli</summary>

### First registration: use the existing code archive

For a function that is not registered yet, the source must be accessible to the
cluster. Digitalhub supports
[S3 ZIP sources](https://scc-digitalhub.github.io/sdk-docs/0.15/reference/configuration/code_src/s3/)
with a `module:callable` handler. Use your existing ZIP containing `main.py`,
`params.yaml`, `requirements.txt`, `lib/`, and any feature lists at its root.
The wrapper folder `oneClick-ML/` must not surround these files inside the ZIP
when using `main:helper_main`.

If the archive is already in platform storage, skip this upload and use its S3
file path directly. Otherwise, upload the existing archive:

```bash
export SOURCE_ARCHIVE='<path-to-existing-code.zip>'
dhcli upload -p "$PROJECT_NAME" artifact \
  -n oneclick_source -f "$SOURCE_ARCHIVE"
dhcli get -p "$PROJECT_NAME" -o yaml artifacts -n oneclick_source
```

Check that the artifact is `READY`. Copy `spec.path` from the YAML output into
`SOURCE_PATH` (the `s3://.../file.zip` address, not the artifact key):

```bash
export SOURCE_PATH='<s3://bucket/path/to/code.zip>'
```

If the code is only in the local folder and has not been archived, this optional
command creates an archive using the `zip` utility. Then upload it as above:

```bash
(
  cd oneClick-ML
  zip -r ../dhcli/oneclick-source.zip main.py params.yaml requirements.txt lib
  if [ -d feature_list_types ]; then
    zip -r ../dhcli/oneclick-source.zip feature_list_types
  fi
)
```

Create the function definition. The Bash loop copies the existing dependency
pins into a YAML list; it does not execute Python or alter the ML code. The CLI
needs dependency values in the definition, rather than a workspace-only path to
`requirements.txt`.

Run this block from the **parent of `oneClick-ML`**. If you are inside
`oneClick-ML`, run `cd ..` first. The subshell stops on errors so registration
does not continue after a failed file operation. It also checks the variables
set in the earlier steps.

```bash
(
set -e
: "${PROJECT_NAME:?Set PROJECT_NAME to your Digitalhub project name}"
: "${SOURCE_PATH:?Set SOURCE_PATH to the uploaded source ZIP S3 path}"
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

Copy the new `ID:` into `FUNCTION_ID` as shown above. The nested
`spec.source.source` and `spec.source.handler` fields represent the SDK's
`code_src` and `handler` arguments; see the version 0.15
[source schema](https://github.com/scc-digitalhub/digitalhub-sdk-runtime-python/blob/0.15.0/digitalhub_runtime_python/entities/function/python/models.py).

</details>

## 5. Build the main helper function

Create `dhcli/build.yaml` with the notebook's build resources:

```bash
cat > dhcli/build.yaml <<'YAML'
spec:
  resources:
    cpu: "4"
    mem: "8Gi"
    disk: "10Gi"
YAML

dhcli run -p "$PROJECT_NAME" -i "$FUNCTION_ID" \
  -f dhcli/build.yaml python+build
dhcli list -p "$PROJECT_NAME" runs
```

In the list, look for `KIND` equal to `python+build:run`. Listing all runs avoids
an encoding problem observed with the `-k` filter: a literal `+` can arrive at
Core as a space (`python build:run`), causing HTTP 400. Shell quotes do not fix
HTTP query encoding. This list error does not establish whether the separately
submitted build succeeded or failed; inspect its run before submitting another.

`dhcli run` submits asynchronously. Unlike the notebook's `wait=True`, command
success only confirms submission. Select the build run belonging to this function
in the run list or Core UI, copy its ID, and inspect it:

```bash
export BUILD_RUN_ID='<build-run-id>'
dhcli get -p "$PROJECT_NAME" -o yaml runs "$BUILD_RUN_ID"
dhcli log -p "$PROJECT_NAME" -f "$BUILD_RUN_ID"
```

Repeat `get` until `status.state` is `COMPLETED`. Do not submit training if the
build fails; inspect the logs and fix the dependency or build error first. Logs
alone do not replace the state check. These commands follow the documented
[run monitoring flow](https://scc-digitalhub.github.io/docs/0.15/cli/usage/#monitor-a-run).

After completion, fetch the function again to get the image reference, replacing
`main_func.refresh().spec.image`:

```bash
dhcli get -p "$PROJECT_NAME" -o yaml functions "$FUNCTION_ID"
```

The [Python build operation](https://scc-digitalhub.github.io/docs/0.15/runtimes/python/)
prepares an image for subsequent executions. Read `spec.image` in the YAML
output and proceed only once it is populated. A previously built function can start directly at step 6
after retrieving its ID and checking its image.

## 6. Run the main helper function on the cluster

Create `dhcli/job.yaml` with the input keys copied in step 3. The unquoted
heredoc delimiter allows Bash to insert the variables into the YAML:

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

dhcli run -p "$PROJECT_NAME" -i "$FUNCTION_ID" \
  -f dhcli/job.yaml python+job
dhcli list -p "$PROJECT_NAME" runs
```

Select the training run with `KIND` equal to `python+job:run`. As with the build,
list without `-k` to avoid the plus-sign encoding issue.

The argument names `csv_file`, `config_file`, and `output_artifact_name` must match
`helper_main`. Artifact references belong in `inputs`; the output name belongs in
`parameters`. `project` is injected by the runtime. See the
[Python job reference](https://scc-digitalhub.github.io/sdk-docs/0.15/reference/runtimes/python/python/python-job/).

Keep the outer `spec` wrapper in both run files: the
[0.15 CLI implementation](https://github.com/scc-digitalhub/digitalhub-cli/blob/0.15.2/handlers/adapter/run.go)
reads additional run settings from that mapping. The CLI resolves function and
task references and submits a remote run, so no separate task creation or local
execution environment is needed.

Copy the training run ID from the list or Core UI and monitor it:

```bash
export JOB_RUN_ID='<training-run-id>'
dhcli get -p "$PROJECT_NAME" -o yaml runs "$JOB_RUN_ID"
dhcli log -p "$PROJECT_NAME" -f "$JOB_RUN_ID"
```

The run also appears in Core's **Runs** section. Wait for `COMPLETED` and inspect
the logs for training errors. The existing helper catches exceptions and prints
`Error during training: ...` without re-raising them, so a completed run alone
does not prove training succeeded. Verify a newly produced output artifact too.

## 7. Inspect and download results

The helper explicitly logs an **artifact**, not a Digitalhub Model entity. Find
`output_data` under **Artifacts** in Core, or use:

```bash
dhcli list -p "$PROJECT_NAME" artifacts -n output_data
dhcli get -p "$PROJECT_NAME" -o yaml artifacts -n output_data
```

Copy the ID of the output artifact produced by your job, then download it:

```bash
export OUTPUT_ID='<output-artifact-id>'
dhcli download -p "$PROJECT_NAME" artifact "$OUTPUT_ID" \
  -d results/output_data
```

Check that the artifact is `READY`, contains the expected results, and was created
by the current training run (check the creation time and run relationship in
Core/metadata). An older `output_data` can remain after a failed training run.
When jobs share an output name, choose the correct artifact version explicitly
from the list rather than assuming the latest one belongs to your run.

## 8. Reuse the pipeline or update its code

- **New terminal session:** select `atlas`, set `PROJECT_NAME`, retrieve the
  artifacts and copy their keys in step 3, then retrieve the function and copy
  its ID in step 4. If its image exists, continue with step 6.
- **Different data or configuration:** upload the changed files in step 3,
  retrieve the intended versions, copy their keys, and recreate `dhcli/job.yaml`.
  Change `output_artifact_name` there to distinguish experiments. Reuse the built
  function when code and dependencies are unchanged.
- **Changed source:** upload the updated archive, set its new `SOURCE_PATH`,
  recreate the function YAML, and register a new function version. Copy its new
  ID, build that version, and submit the job with that same ID. Rebuild whenever
  dependencies change.

For the notebook's code-only image-reuse option, retrieve the old function in
YAML format and copy `spec.image`. Add `image: "<existing-image-reference>"`
under `spec` in the new function YAML, alongside `python_version`, before
creating the new version. Keep the new source address. Then retrieve the new
function's ID and proceed to step 6. Use this shortcut only when the existing
image contains all required dependencies; otherwise build the new version.

The CLI's create/get/list/run/log/download syntax is documented in the
[Digitalhub 0.15 command reference](https://scc-digitalhub.github.io/docs/0.15/cli/commands/).
